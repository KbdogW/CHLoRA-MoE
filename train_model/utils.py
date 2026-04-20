from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.utils import class_weight

""" Trainer Class """  
class WeightedTrainer(Trainer):
    # def __init__(self, class_weights, **kwargs):
    #     super().__init__(**kwargs)
    #     self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # # 1. 获取 labels
        # labels = inputs.get("labels")
        
        # # 2. 前向传播
        # outputs = model(**inputs)
        
        # # 3. 获取 logits
        # logits = outputs.get("logits")
        
        # # 4. === 关键修复：获取 num_labels ===
        # # 尝试直接从模型获取，如果被 DDP 包裹(module)，则多一层获取
        # if hasattr(model, "num_labels"):
        #     num_labels = model.num_labels
        # elif hasattr(model, "module") and hasattr(model.module, "num_labels"):
        #     num_labels = model.module.num_labels
        # else:
        #     # 如果实在找不到，回退到 config (兼容标准 HF 模型) 或者默认值
        #     num_labels = getattr(model.config, "num_labels", 2)
        # # 5. === 关键修复：加权 Loss 计算 ===
        # # 确保权重在正确的设备上
        # if self.class_weights.device != logits.device:
        #     self.class_weights = self.class_weights.to(logits.device)
            
        # loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # # 计算 Loss (使用修正后的 num_labels)
        # loss_det = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        # # 6. Router Loss (如果有)
        # if "router_logits" in outputs and "severity_labels" in inputs:
        #      loss_router_fct = nn.CrossEntropyLoss() 
        #      loss_router = loss_router_fct(outputs["router_logits"], inputs["severity_labels"])
        #      loss = loss_det + 0.5 * loss_router
        # else:
        #      loss = loss_det
        outputs = model(**inputs)
        
        # 这里的 loss 已经是 moe.py 里算好的、加权过的总 Loss
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


""" Define training arguments """ 
def define_training_args(
    output_dir, 
    batch_size, 
    num_steps=500, 
    lr=1.0e-4, 
    gradient_accumulation_steps=1, 
    warmup_steps=500
    ): 
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=False,
        max_steps=num_steps,
        warmup_steps=warmup_steps,
        logging_steps=100,
        eval_steps=num_steps//10,
        save_steps=num_steps//10,
        ddp_find_unused_parameters=True, 
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", 
        greater_is_better=True,
        fp16=False,
        fp16_full_eval=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        weight_decay=0.01)
    return training_args


""" Define Class Weights """
def compute_class_weights(df_train, device):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(df_train["label"]),
        y=np.array(df_train["label"])
    )
    # === [手动干预] ===
    # class_weights[0] = class_weights[0] * 0.8
    class_weights[2] = class_weights[2] * 1.5  # Moderate
    class_weights[3] = class_weights[3] * 1.5  # Severe
    # class_weights[4] = class_weights[4] * 1.5  # Severe

    # class_weights = torch.tensor(class_weights, device="cuda", dtype=torch.float32)
    class_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
    print(f"Final Adjusted Class Weights: {class_weights}")
    return class_weights


""" 辅助函数：解包数据 """
def unwrap_data(data):
    # 如果是元组 (Tuple)，通常第一个元素是我们需要的 (logits 或 labels)
    if isinstance(data, tuple):
        return data[0]
    return data

# """ Define Metric (Multi-class) """
# def compute_metrics(pred):
#     # 1. 解包 Labels
#     # Trainer 可能会把 (labels, severity_labels) 打包在一起返回
#     labels = unwrap_data(pred.label_ids)
    
#     # 2. 解包 Predictions
#     # Model 返回 (logits, router_logits)
#     predictions = unwrap_data(pred.predictions)
        
#     preds = np.argmax(predictions, axis=1)
#     acc = accuracy_score(labels, preds)
#     f1_macro = f1_score(labels, preds, average='macro')
    
#     # Top-3 logic
#     if predictions.shape[1] > 2:
#         preds_top3 = np.argsort(predictions, axis=1)[:,-3:]
#         preds_top3 = np.array([preds_top3[i] for i in range(len(labels)) if labels[i] in preds_top3[i]])
#         acc_top3 = len(preds_top3) / len(labels)
#     else:
#         acc_top3 = acc 

#     print('Accuracy: ', acc)
#     print('F1 Macro: ', f1_macro)
#     print("Confusion Matrix:\n", confusion_matrix(labels, preds))
#     return { 
#         'accuracy': acc, 
#         'f1_macro': f1_macro, 
#         'top3_accuracy': acc_top3
#         }

""" Define Metric (Multi-class 顶会标准版) """
def compute_metrics(pred):
    # 1. 解包 Labels 和 Predictions
    labels = unwrap_data(pred.label_ids)
    predictions = unwrap_data(pred.predictions)
        
    preds = np.argmax(predictions, axis=1)
    
    # 2. 计算基础指标
    acc = accuracy_score(labels, preds)
    
    # 3. 计算 Macro 指标 (加入 zero_division=0 防止某类别未被预测时报警告)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Top-3 logic (如果在 5 分类及以上任务中比较有用)
    if predictions.shape[1] >= 3:
        preds_top3 = np.argsort(predictions, axis=1)[:, -3:]
        # 统计真实标签是否在 top3 预测中
        hits = sum([1 for i in range(len(labels)) if labels[i] in preds_top3[i]])
        acc_top3 = hits / len(labels)
    else:
        acc_top3 = acc 

    # 4. 打印更清晰的日志表格
    print('\n' + '='*40)
    print(f"{'Metric':<15} | {'Value'}")
    print('-'*40)
    print(f"{'Accuracy':<15} | {acc:.4f}")
    print(f"{'Precision (M)':<15} | {precision_macro:.4f}")
    print(f"{'Recall (M)':<15} | {recall_macro:.4f}")
    print(f"{'F1 Macro':<15} | {f1_macro:.4f}")
    print(f"{'Top-3 Acc':<15} | {acc_top3:.4f}")
    print('-'*40)
    print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    print('='*40 + '\n')
    
    # 5. 返回给 Trainer
    # 注意：这里的 key 名字会决定终端里 eval_xxx 的名字
    return { 
        'accuracy': acc, 
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro, 
        'top3_accuracy': acc_top3
    }
""" Define Metric (Binary) """
# def compute_metrics_binary(pred):
#     # 1. 解包 Labels
#     # 可能会收到 (labels, severity_labels) 的元组，我们只取第一个
#     labels = unwrap_data(pred.label_ids)
    
#     # 2. 解包 Predictions
#     # 可能会收到 (logits, router_logits) 的元组，我们只取第一个
#     predictions = unwrap_data(pred.predictions)
    
#     preds = np.argmax(predictions, axis=1)
#     acc = accuracy_score(labels, preds)
#     f1_macro = f1_score(labels, preds, average='macro')
    
#     # 计算 AUC
#     probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
    
#     try:
#         if predictions.shape[1] == 2:
#             auc = roc_auc_score(labels, probs[:, 1])
#         else:
#             auc = 0.0 
#     except ValueError:
#         auc = 0.0

#     print('\nAUC: ', auc)
#     print('Accuracy: ', acc)
#     print('F1 Macro: ', f1_macro)
#     print("Confusion Matrix:\n", confusion_matrix(labels, preds))
    
#     return { 
#         'accuracy': acc, 
#         'f1_macro': f1_macro, 
#         'auc': auc 
#         }
def compute_metrics_binary(pred):
    labels = unwrap_data(pred.label_ids)
    predictions = unwrap_data(pred.predictions)
    
    # 获取预测为 Class 1 的概率
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()[:, 1]
    
    # --- [新增] 动态寻找最佳阈值 ---
    best_f1 = 0
    best_threshold = 0.5
    
    # 遍历 0.1 到 0.9 的阈值
    for threshold in np.arange(0.1, 0.9, 0.05):
        temp_preds = (probs > threshold).astype(int)
        f1 = f1_score(labels, temp_preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"\nBest Threshold: {best_threshold}, Best F1: {best_f1}")
    
    # 使用最佳阈值生成最终预测
    final_preds = (probs > best_threshold).astype(int)
    acc = accuracy_score(labels, final_preds)
    f1_macro = f1_score(labels, final_preds, average='macro')
    auc = roc_auc_score(labels, probs) # AUC 不受阈值影响
    
    print('AUC: ', auc)
    print('Accuracy: ', acc)
    print('F1 Macro: ', f1_macro)
    print("Confusion Matrix (Optimized):\n", confusion_matrix(labels, final_preds))
    
    return { 
        'accuracy': acc, 
        'f1_macro': f1_macro, 
        'auc': auc 
    }