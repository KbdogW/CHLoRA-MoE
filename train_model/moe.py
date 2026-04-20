import torch
import torch.nn as nn
from transformers import WavLMModel, HubertModel
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import torch.nn.functional as F
from transformers.models.wavlm.modeling_wavlm import WavLMEncoder

class FocalLoss(nn.Module):
    """
    多分类 Focal Loss
    结合了 Class Weights (alpha) 和 Focusing Parameter (gamma)
    """
    def __init__(self, weight=None, gamma=2.5, label_smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight # 类别权重 alpha
        self.gamma = gamma   # 聚焦参数，通常设为 2.0
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. 计算标准的交叉熵损失 (支持标签平滑)
        # reduction='none' 确保我们拿到每个样本单独的 loss
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.weight, 
            label_smoothing=self.label_smoothing, 
            reduction='none'
        )
        
        # 2. 计算 pt (模型对真实类别的预测概率)
        # 因为 ce_loss = -log(pt)，所以 pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. 计算 Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. 归一化
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SelfAttentionPooling(nn.Module):
    """
    自注意力池化层：
    自动学习时间步的权重，重点关注包含病理特征的片段，忽略静音和噪声。
    """
    def __init__(self, input_dim):
        super().__init__()
        # 这是一个简单的线性层，将特征映射到一个标量权重
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh() # 激活函数，增加非线性
        )

    def forward(self, x, attention_mask=None):
        # x shape: [Batch, Time, Dim]
        
        # 1. 计算注意力分数 [Batch, Time, 1]
        attn_scores = self.attention(x) 
        
        # 2. 处理 Padding (非常重要！防止 Padding 参与计算)
        if attention_mask is not None:
            # 扩展 mask 维度以匹配 scores
            # attention_mask通常是 [Batch, Time], 1为有效，0为padding
            extended_mask = attention_mask.unsqueeze(-1) 
            # 将 padding 位置的分数设为负无穷，这样 softmax 后权重为 0
            attn_scores = attn_scores.masked_fill(extended_mask == 0, -1e9)
        
        # 3. 计算权重 [Batch, Time, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 4. 加权求和 [Batch, Dim]
        # Sum( Weights * Features )
        pooled_output = torch.sum(x * attn_weights, dim=1)
        
        return pooled_output

class PhysicianGuidedMoEWavLM(nn.Module):
    def __init__(self, base_model_name, num_labels, num_experts=5, class_weights=None, router_weights=None):
        super().__init__()
        self.num_labels = num_labels
        self.num_experts = num_experts # 4: Healthy, Mild, Moderate, Severe
        
        # === [关键] 注册 class_weights ===
        # 使用 register_buffer 可以确保它自动随模型移动到 GPU，且不作为可训练参数
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None   
        # === [新增] 注册 Router 权重 ===
        if router_weights is not None:
            self.register_buffer('router_weights', router_weights)
        else:
            self.router_weights = None # 没传就为 None 
        print(f"Weights Configured -> Diagnosis: {self.class_weights}, Router: {self.router_weights}")

        # 1. 加载冻结的 Backbone (WavLM)
        print(f"Loading Backbone: {base_model_name}")
        self.wavlm = WavLMModel.from_pretrained(base_model_name, local_files_only=True)
        # 冻结所有参数
        for param in self.wavlm.parameters():
            param.requires_grad = False
           
        # 2. 定义分层 LoRA 配置
        # Shared LoRA: 作用于底层 (例如 0-5 层)
        self.shared_config = LoraConfig(
            r=16, lora_alpha=32, #8,16
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "intermediate_dense", "output_dense"],
            layers_to_transform=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # 只改底层
            lora_dropout=0.3, bias="none"
        )
        
        # Expert LoRA: 作用于高层 (例如 6-11 层)
        self.expert_config = LoraConfig(
            r=16, lora_alpha=32, # 专家层参数可以多一点
            target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "intermediate_dense", "output_dense"],
            layers_to_transform=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], # 只改高层
            lora_dropout=0.3, bias="none"
        )

        # 3. 注入 Adapters
        # 我们使用 peft 的 add_adapter 机制
        # 首先把 model 变成支持 peft 的状态
        self.wavlm = get_peft_model(self.wavlm, self.shared_config, adapter_name="shared_bottom")
        
        # 添加 4 个专家 Adapter
        # expert_names = ["expert_healthy", "expert_mild", "expert_moderate", "expert_severe"]
        # self.expert_names = expert_names
        # 🔴【核心修改】动态生成专家名称，适配任意类别数量 (3类、4类、5类通用)
        self.expert_names = [f"expert_{i}" for i in range(self.num_experts)]
        for name in self.expert_names:
            self.wavlm.add_adapter(name, self.expert_config)

        # 4. 定义 Router (分类器)
        # 输入是 WavLM 的 hidden_size (Base=768, Large=1024)
        hidden_size = self.wavlm.config.hidden_size
        # 添加注意力池化层
        # self.attention_pool = nn.Sequential(
        #     nn.Linear(hidden_size, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 1)  # 输出每个时间步的重要性分数
        # )
        self.router_pooling = SelfAttentionPooling(hidden_size)
        self.router = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts) # 输出 4 个专家的 Logits
        )

        # 5. 定义最终的分类头 (Detection Head)
        # 这是一个简单的 Projection Head
        self.final_pooling = SelfAttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels) # 输出 s/p (2类)
        )
        # =============== 【关键修复】 ===============
        # PEFT 的 add_adapter 默认可能不会开启梯度，或者因为当前未激活而被视为冻结。
        # 我们需要强制开启所有 LoRA 参数的梯度，确保 Optimizer 能看到它们。
        
        # 1. 先打印确认一下当前状态
        print("Unfreezing all LoRA parameters...")
        
        # 2. 暴力开启所有包含 'lora' 字段的参数的梯度
        # for name, param in self.wavlm.named_parameters():
        #     if "lora" in name:
        #         param.requires_grad = True
        # # 在 moe.py 的 __init__ 中，冻结完参数后加入：解冻所有的 LayerNorm 层
        # for name, param in self.wavlm.named_parameters():
        #     if "layer_norm" in name:
        #         param.requires_grad = True 
        trainable_keywords = ["lora", "layer_norm", "feature_extractor"]
        for name, param in self.wavlm.named_parameters():
            if any(k in name for k in trainable_keywords):
                param.requires_grad = True        
        
        # 3. 再次确认 Router 和 Classifier 也是开启的
        for param in self.router.parameters():
            param.requires_grad = True
        # for param in self.attention_pool.parameters():  # 新增这一行
        #     param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        # ===========================================
        #self.register_buffer('class_weights', class_weights) # 将权重注册为 buffer，随模型保存
        print(f"class_weights type: {type(class_weights)}")
        print(f"class_weights value: {class_weights}")
   
    
    # 消除 for 循环
    def forward(self, input_values, labels=None, severity_labels=None, attention_mask=None, **kwargs):
        batch_size = input_values.shape[0]
        
        # 1. 激活 Shared Bottom 跑一次 (获取底层特征用于路由)
        self.wavlm.set_adapter("shared_bottom")
        outputs_shared = self.wavlm(input_values, attention_mask=attention_mask, output_hidden_states=True)
        middle_hidden_state = outputs_shared.hidden_states[10] # 取中间层
        
        # 2. Router 计算
        router_input = self.router_pooling(middle_hidden_state, attention_mask)
        router_logits = self.router(router_input)
        
        # 计算负载均衡 Loss
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        mean_router_probs = torch.mean(router_probs, dim=0)
        target_dist = torch.full_like(mean_router_probs, 1.0 / self.num_experts)
        loss_load_balance = torch.nn.functional.mse_loss(mean_router_probs, target_dist)

        # 3. 确定每个样本属于哪个专家
        if self.training and severity_labels is not None:
             target_indices = severity_labels 
        else:
             target_indices = torch.argmax(router_logits, dim=-1)

        # === 核心优化：按专家分组，而不是按样本循环 ===
        # 准备一个容器来存最终的 logits，顺序必须和 input 一致
        final_logits = torch.zeros(batch_size, self.num_labels, device=input_values.device)
        
        # 遍历 4 个专家
        for expert_idx, expert_name in enumerate(self.expert_names):
            # 找出当前 batch 中被分配给该专家的样本索引
            # indices_for_this_expert: 例如 [0, 3, 5] 表示第0,3,5个样本属于该专家
            indices_for_this_expert = (target_indices == expert_idx).nonzero(as_tuple=True)[0]
            
            if len(indices_for_this_expert) == 0:
                continue # 如果没有样本选这个专家，跳过
            
            # 激活该专家 Adapter
            self.wavlm.set_adapter(expert_name)
            
            # 提取属于该专家的子 Batch
            sub_input = input_values[indices_for_this_expert]
            sub_mask = attention_mask[indices_for_this_expert] if attention_mask is not None else None
            
            # 跑一次 WavLM (并行处理多个样本！)
            out_expert = self.wavlm(sub_input, attention_mask=sub_mask)
            
            # 提取特征并分类
            feat = self.final_pooling(out_expert.last_hidden_state, sub_mask)
            sub_logits = self.classifier(feat)
            
            # 将结果填回对应的位置
            final_logits[indices_for_this_expert] = sub_logits

        # # --- Step 3: 计算 Loss (保持不变) ---
        # loss = None
        # if labels is not None:
        #     # ... (这部分 Loss 计算代码保持不变) ...
        #     if self.class_weights is not None:
        #         loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
        #     else:
        #         loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
        #     loss_det = loss_fct(final_logits, labels)
            
        #     if self.router_weights is not None:
        #         loss_router_fct = nn.CrossEntropyLoss(weight=self.router_weights, label_smoothing=0.1)
        #     else:
        #         loss_router_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
        #     loss_router = loss_router_fct(router_logits, severity_labels)
            
        #     loss = loss_det + 0.3 * loss_router + 0.1 * loss_load_balance
        
        # --- Step 3: 计算 Loss (替换为 Focal Loss) ---
        loss = None
        if labels is not None:
            # 1. 诊断 Loss (主任务) - 增大 gamma 加强对困难样本的关注
            loss_fct = FocalLoss(
                weight=self.class_weights, 
                gamma=2.5, # 推荐起点是 2.0
                label_smoothing=0.1
            )
            loss_det = loss_fct(final_logits, labels)
            
            # 2. 路由 Loss (辅助任务) - Router 同样面临严重的类别不平衡
            loss_router_fct = FocalLoss(
                weight=self.router_weights, 
                gamma=2.5, 
                label_smoothing=0.1
            )
            loss_router = loss_router_fct(router_logits, severity_labels)
            
            # 总 Loss 计算不变
            loss = loss_det + 0.3 * loss_router + 0.1 * loss_load_balance
        
        return {
            "loss": loss,
            "logits": final_logits,
            "router_logits": router_logits
        }

