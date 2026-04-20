# -*- coding: utf-8 -*-
"""
export_onnx.py - MoE 语音分类模型 ONNX 静态化导出脚本

功能:
1. 加载基于 PEFT 的 MoE 模型
2. 提取 Expert LoRA 权重，并合并 Shared Bottom LoRA
3. 通过 Monkey Patching 将高层 Linear 替换为支持 Soft Routing 的自定义层
4. 使用 Hook 机制拦截中间层输出，计算 Router 概率
5. 导出支持动态 Batch Size 和 Sequence Length 的 ONNX 模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor

from moe import PhysicianGuidedMoEWavLM

# =============================================================================
# 自定义 Soft Routing LoRA 线性层
# =============================================================================
class SoftRoutedLoRALinear(nn.Module):
    """
    替换目标 Linear 层，执行 Dense Execution。
    结合了基础权重和多个专家的 LoRA 权重。
    """
    def __init__(self, base_linear: nn.Linear, expert_weights_dict: dict, scaling: float, shared_state: dict):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        
        # 继承基础权重
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.scaling = scaling
        
        self.num_experts = len(expert_weights_dict)
        
        # 注册专家权重为 buffer，使其成为 ONNX 计算图的一部分
        for i in range(self.num_experts):
            self.register_buffer(f'lora_A_{i}', expert_weights_dict[i]['A'])
            self.register_buffer(f'lora_B_{i}', expert_weights_dict[i]['B'])
            
        # 使用共享状态字典，避免循环引用导致 RecursionError
        self.shared_state = shared_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 基础线性层计算
        base_out = F.linear(x, self.weight, self.bias)
        
        # 如果未设置 router_probs（防呆），直接返回基础结果
        if 'router_probs' not in self.shared_state:
            return base_out
            
        # [B, num_experts]
        router_probs = self.shared_state['router_probs']
        
        # 2. 软路由 (Dense Execution)
        delta_out = torch.zeros_like(base_out)
        for i in range(self.num_experts):
            weight_A = getattr(self, f'lora_A_{i}') # [r, in_features]
            weight_B = getattr(self, f'lora_B_{i}') # [out_features, r]
            
            # x: [B, T, in_features]
            # PyTorch F.linear 计算: x @ weight.T
            # 因此这里相当于 x @ A.T @ B.T
            expert_out = F.linear(x, weight_A)
            expert_out = F.linear(expert_out, weight_B)
            expert_out = expert_out * self.scaling
            
            # 按 Router 概率加权: prob [B, 1, 1]
            prob = router_probs[:, i].view(-1, 1, 1)
            delta_out = delta_out + prob * expert_out
            
        return base_out + delta_out

# =============================================================================
# ONNX 导出包装器
# =============================================================================
class ExportMoEWrapper(nn.Module):
    def __init__(self, original_model: PhysicianGuidedMoEWavLM):
        super().__init__()
        
        self.shared_state = {}
        
        # ==========================================
        # 1. 提取 Expert LoRA 权重 (必须在合并前进行)
        # ==========================================
        print("正在提取 Expert LoRA 权重...")
        self.expert_lora_weights = {i: {} for i in range(original_model.num_experts)}
        peft_model = original_model.wavlm
        
        # 专家作用的目标模块
        target_modules = ["k_proj", "v_proj", "q_proj", "out_proj", "intermediate_dense", "output_dense"]
        expert_layers = range(10, 24) # 10~23 层
        
        for name, param in peft_model.named_parameters():
            if "lora_" not in name:
                continue
                
            for expert_idx in range(original_model.num_experts):
                expert_name = f"expert_{expert_idx}"
                if expert_name in name:
                    # 确定这是属于哪个 layer 和哪个 module 的
                    # name e.g.: base_model.model.encoder.layers.10.attention.q_proj.lora_A.expert_0.weight
                    parts = name.split('.')
                    layer_idx = -1
                    module_name = ""
                    for p_idx, p in enumerate(parts):
                        if p.isdigit():
                            layer_idx = int(p)
                        if p in target_modules:
                            module_name = p
                            
                    if layer_idx in expert_layers and module_name:
                        key = f"layers.{layer_idx}.{module_name}"
                        if key not in self.expert_lora_weights[expert_idx]:
                            self.expert_lora_weights[expert_idx][key] = {}
                            
                        if "lora_A" in name:
                            self.expert_lora_weights[expert_idx][key]['A'] = param.data.clone()
                        elif "lora_B" in name:
                            self.expert_lora_weights[expert_idx][key]['B'] = param.data.clone()
                            
        # ==========================================
        # 2. 合并 Shared Bottom LoRA 并卸载 PEFT 结构
        # ==========================================
        print("正在合并 Shared Bottom LoRA 并移除动态 Adapter 架构...")
        peft_model.set_adapter("shared_bottom")
        merged_wavlm = peft_model.merge_and_unload()
        
        self.wavlm = merged_wavlm
        self.router_pooling = original_model.router_pooling
        self.router = original_model.router
        self.final_pooling = original_model.final_pooling
        self.classifier = original_model.classifier
        
        # ==========================================
        # 3. 进行“深度手术” (Monkey Patching)
        # ==========================================
        print("正在替换目标层的线性模块，应用 SoftRoutedLoRALinear...")
        scaling = 32.0 / 16.0 # lora_alpha / r
        patched_count = 0
        
        for layer_idx in expert_layers:
            layer_module = self.wavlm.encoder.layers[layer_idx]
            
            # 需要替换的模块映射字典 (属性名 -> 原模块)
            modules_to_patch = {
                'attention.q_proj': layer_module.attention.q_proj,
                'attention.k_proj': layer_module.attention.k_proj,
                'attention.v_proj': layer_module.attention.v_proj,
                'attention.out_proj': layer_module.attention.out_proj,
                'feed_forward.intermediate_dense': layer_module.feed_forward.intermediate_dense,
                'feed_forward.output_dense': layer_module.feed_forward.output_dense
            }
            
            for module_path, original_linear in modules_to_patch.items():
                module_name = module_path.split('.')[-1]
                key = f"layers.{layer_idx}.{module_name}"
                
                # 收集所有专家在当前模块的 A/B 矩阵
                expert_dict = {}
                for i in range(original_model.num_experts):
                    if key in self.expert_lora_weights[i]:
                        expert_dict[i] = self.expert_lora_weights[i][key]
                
                if len(expert_dict) == original_model.num_experts:
                    # 创建新层
                    soft_layer = SoftRoutedLoRALinear(original_linear, expert_dict, scaling, self.shared_state)
                    
                    # 替换原模块
                    parent_name, child_name = module_path.split('.')
                    parent_module = getattr(layer_module, parent_name)
                    setattr(parent_module, child_name, soft_layer)
                    patched_count += 1
                    
        print(f"成功注入了 {patched_count} 个 SoftRoutedLoRALinear 模块。")

        # ==========================================
        # 4. 注册 Hook 拦截第 9 层输出
        # ==========================================
        print("注册 Forward Hook 拦截 Router 输入...")
        # 注意: hidden_states[10] 实际上对应 encoder.layers[9] 的输出
        target_hook_layer = self.wavlm.encoder.layers[9]
        
        def router_hook(module, input, output):
            # output 是一个 tuple: (hidden_states, ...)
            hidden_states = output[0]
            
            # 计算 Router 概率
            router_input = self.router_pooling(hidden_states)
            router_logits = self.router(router_input)
            
            # 保存到 shared_state 中，供后续被 patch 的层使用
            self.shared_state['router_logits'] = router_logits
            self.shared_state['router_probs'] = torch.softmax(router_logits, dim=-1)
            
        # 保存 hook 句柄，虽然我们只是导出一次，这是一种好习惯
        self.hook_handle = target_hook_layer.register_forward_hook(router_hook)

    def forward(self, input_values: torch.Tensor):
        """
        全静态的前向传播: 
        没有 if/else，没有 set_adapter，没有 argmax 索引。
        """
        # 每次 forward 时清空状态，防止数据残留
        self.shared_state.clear()
        
        # 1. 一次性走通主干网络
        # 当流经 layer[9] 时，hook 会自动触发并计算 self.shared_state['router_probs']
        # 当流经 layer[10-23] 时，SoftRoutedLoRALinear 会自动获取概率并执行 Dense Execution
        outputs = self.wavlm(input_values, output_hidden_states=False)
        last_hidden_state = outputs.last_hidden_state
        
        # 2. 提取特征并进行最终分类
        feat = self.final_pooling(last_hidden_state)
        logits = self.classifier(feat)
        
        # 返回分类 logits 和 严重程度的 logits (可选用于辅助输出)
        return logits, self.shared_state.get('router_logits', torch.zeros(1, len(self.expert_lora_weights), device=logits.device))


# =============================================================================
# 主导出逻辑
# =============================================================================
def export_to_onnx(checkpoint_path: str, output_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备: {device}")
    
    # 1. 加载原生模型
    print(f"加载原生 PyTorch 模型及权重...")
    original_model = PhysicianGuidedMoEWavLM(
        base_model_name="microsoft/wavlm-large",
        num_labels=5,
        num_experts=5
    )
    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    state_dict = load_file(weights_path)
    original_model.load_state_dict(state_dict, strict=False)
    original_model.eval()
    
    # 2. 包装为静态化模型
    print("\n" + "="*40)
    print("开始模型深度手术转换...")
    print("="*40)
    static_model = ExportMoEWrapper(original_model)
    static_model.to(device)
    static_model.eval()
    
    # 3. 准备 Dummy Input (批大小 1，长度 48000)
    # Triton 部署需要支持可变批次和可能存在的轻微长度变化，使用动态轴
    dummy_input = torch.randn(1, 48000, device=device)
    
    # 4. 执行导出
    print("\n" + "="*40)
    print(f"开始导出 ONNX 模型至: {output_path} ...")
    print("="*40)
    
    torch.onnx.export(
        static_model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=16, # 使用较新的 opset 支持更多的张量操作
        do_constant_folding=True,
        input_names=['input_values'],
        output_names=['logits', 'router_logits'],
        dynamic_axes={
            'input_values': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'},
            'router_logits': {0: 'batch_size'}
        }
    )
    
    print("✅ ONNX 导出完成！")
    print(f"请使用 Triton 等工具验证生成的 {output_path}。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="导出 MoE 模型为 ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoint-9800", help="模型权重目录")
    parser.add_argument("--output", type=str, default="model.onnx", help="导出的 ONNX 文件路径")
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output)
