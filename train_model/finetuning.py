# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.distributed as dist
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2FeatureExtractor
import pandas as pd
import argparse
import random
import numpy as np
from sklearn.utils import class_weight

import warnings
warnings.filterwarnings("ignore")

from moe import PhysicianGuidedMoEWavLM 
from dataset import Dataset
from utils import WeightedTrainer, define_training_args, \
    compute_metrics, compute_class_weights, compute_metrics_binary
    

""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="vocal")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=16,
        type=int,
        required=False)
    parser.add_argument(
        "--steps",
        help="number of training steps",
        default=300, ## 2500 for category
        type=int,
        required=False)
    parser.add_argument(
        "--feature_extractor",
        help="feature extractor to use",
        default="facebook/hubert-base-ls960",  
        type=str,                          
        required=False) 
    parser.add_argument(
        "--model",
        help="model to use",
        default="facebook/hubert-base-ls960",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_train",
        help="path to the train df",
        default="data/df_train.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--df_val",
        help="path to the val df",
        default="data/df_val.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--df_test",
        help="path to the test df",
        default="data/df_test.csv",
        type=str,
        required=False)  
    parser.add_argument(
        "--save_confidence_scores",
        help="whether to save confidence scores or not",
        action="store_true",
        required=False)
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results",
        type=str,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=10,
        type=int,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum audio duration",
        default=3.0,
        type=float,
        required=False)
    parser.add_argument(
        "--label",
        help="Label to predict; choose one from ['s/p', 'category', 'macro-category']",
        default="category",
        type=str,
        required=False)
    parser.add_argument(
        "--augmentation",
        help="whether to augment or not the data",
        action="store_true",
        required=False)
    parser.add_argument(
        '--seed',
        help='Random seed for training',
        type=int,  
        required=True)
        
    args = parser.parse_args()
    return args


""" Read and Process Data"""
def read_data(df_train_path, df_val_path, label_name):
    df_train = pd.read_csv(df_train_path, index_col=None)
    df_val = pd.read_csv(df_val_path, index_col=None)
    #severity_map = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    ## Prepare Labels
    severity_map = {"0": 0, "High": 1, "Mid": 2, "Low": 3, "Very Low": 4}
    if 'severity' in df_train.columns and df_train['severity'].dtype == object:
        df_train['severity'] = df_train['severity'].map(severity_map)
        df_val['severity'] = df_val['severity'].map(severity_map)
    #labels = df_train[label_name].unique()
    # 强制按 0, 1, 2, 3 排序
    labels = sorted(df_train[label_name].unique())

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Train
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index,label_name]]
    df_train['label'] = df_train['label'].astype(int)

    ## Validation
    for index in range(0,len(df_val)):
        df_val.loc[index,'label'] = label2id[df_val.loc[index,label_name]]
    df_val['label'] = df_val['label'].astype(int)

    print("Label2Id: ", label2id)
    print("Id2Label: ", id2label)
    print("Num Labels: ", num_labels)

    return df_train, df_val, num_labels, label2id, id2label, labels

""" Define model and feature extractor """
def define_model(
    model_checkpoint, 
    feature_extractor_checkpoint, 
    num_labels, 
    label2id, 
    id2label, 
    device="cuda", 
    class_weights=None,
    router_weights=None
    ):
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_checkpoint)
    print(f"Initializing Physician-Guided MoE Model based on {model_checkpoint}...")
    # 实例化自定义模型
    # 注意：这里不再用 AutoModelForAudioClassification
    model = PhysicianGuidedMoEWavLM(
        base_model_name=model_checkpoint,
        num_labels=num_labels,
        num_experts=5,  # 对应 Healthy, Mild, Moderate, Severe
        class_weights=class_weights,    # 传入权重
        router_weights=router_weights 
    ).to(device)
    return feature_extractor, model

def print_lora_parameters(model):
    """专门为LoRA微调设计的参数打印"""
    
    print("\n" + "=" * 80)
    print("LoRA TRAINABLE PARAMETERS")
    print("=" * 80)
    
    lora_params = []
    non_lora_trainable = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora' in name.lower():
                lora_params.append((name, param))
            else:
                non_lora_trainable.append((name, param))
        else:
            frozen_params.append((name, param))
    
    # 打印LoRA参数
    # if lora_params:
    #     print("\nLoRA parameters:")
    #     print("-" * 40)
    #     for name, param in sorted(lora_params):
    #         print(f"  {name:70} | shape: {tuple(param.shape)}")
        
    #     total_lora = sum(p.numel() for _, p in lora_params)
    #     print(f"\n  Total LoRA parameters: {total_lora:,}")
    
    # 打印其他可训练参数
    if non_lora_trainable:
        print("\nOther trainable parameters (non-LoRA):")
        print("-" * 40)
        for name, param in sorted(non_lora_trainable)[:10]:  # 只显示前10个
            print(f"  {name:70} | shape: {tuple(param.shape)}")
        
        if len(non_lora_trainable) > 10:
            print(f"  ... and {len(non_lora_trainable) - 10} more")
        
        total_lora = sum(p.numel() for _, p in lora_params)
        total_non_lora = sum(p.numel() for _, p in non_lora_trainable)
        print(f"\n  Total non-LoRA trainable parameters: {total_non_lora:,}")
    
    # 计算总数
    total_trainable = total_lora + total_non_lora if 'total_lora' in locals() and 'total_non_lora' in locals() else 0
    total_frozen = sum(p.numel() for _, p in frozen_params)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    if 'total_lora' in locals():
        print(f"  LoRA parameters:        {total_lora:12,}")
    if 'total_non_lora' in locals():
        print(f"  Other trainable:        {total_non_lora:12,}")
    print(f"  Total trainable:        {total_trainable:12,}")
    print(f"  Frozen parameters:      {total_frozen:12,}")
    print(f"  Total parameters:       {total_trainable + total_frozen:12,}")
    print(f"  Trainable percentage:   {100 * total_trainable / (total_trainable + total_frozen):.4f}%")

""" Main Program """
if __name__ == '__main__':

    ## Utils 
    args = parse_cmd_line_params()
    # --- 原代码 ---
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device: ", device)
    # --- 修改为 ---
    # 获取当前进程的 local_rank (由 torchrun 自动设置环境变量)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 只在主进程打印
    if local_rank in [-1, 0]:
        print(f"Device: {device} (Local Rank: {local_rank})")
    print("------------------------------------")
    print("Running with the following parameters:")
    print("Batch size: ", args.batch)
    print("Number of steps: ", args.steps)
    print("Model: ", args.model)
    print("Warmup steps: ", args.warmup_steps)
    print("Learning rate: ", args.lr)
    print("Maximum audio duration: ", args.max_duration)
    print("Label to predict: ", args.label)
    print("------------------------------------\n")

    

    ## Set seed 
    seed = args.seed
    print("Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    output_dir = os.path.join(args.output_dir, str(seed))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir) 
        
    ## Train & Validation df
    df_train, df_val, num_labels, label2id, id2label, labels = read_data(
        args.df_train, 
        args.df_val, 
        args.label
        )
    print("Num labels: ", num_labels)

    class_weights = compute_class_weights(df_train, device) # 确保这一步在 define_model 之前做 
    print(f"Computed Class Weights: {class_weights}")
    # === [新增] 计算 Router Weights (严重程度 4 分类) ===
    df_train['severity_idx'] = df_train['severity']
    # 2. 计算权重
    # classes=[0,1,2,3] 确保生成的权重向量是按顺序对应 0-3 的
    router_weights_np = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2, 3, 4]), 
        y=df_train['severity_idx'].values
    )
    # 3. 转为 Tensor 并移动到 device
    router_weights = torch.tensor(router_weights_np, dtype=torch.float32).to(device)
    #router_weights = compute_class_weights(df_train)    #改为和class_weights权重相同
    
    print(f"Computed Router Weights: {router_weights}")
    # ==================================================
    ## Model & Feature Extractor
    model_checkpoint = args.model
    model_name = model_checkpoint.split("/")[-1]
       
    if 'label2id' in locals():
        label2id = {str(k): int(v) for k, v in label2id.items()}
    if 'id2label' in locals():
        id2label = {int(k): str(v) for k, v in id2label.items()}
 
    feature_extractor, model = define_model(
        model_checkpoint, 
        args.feature_extractor, 
        num_labels, 
        label2id, 
        id2label, 
        device,
        class_weights=class_weights,
        router_weights=router_weights
        )

    # # 新增：分布式训练优化（核心）
    # if torch.distributed.is_initialized():
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model,
    #         device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
    #         find_unused_parameters=True,  # 关闭冗余参数检查，解决日志警告
    #         broadcast_buffers=False  # 关闭缓冲区广播，减轻通信压力
    #     )

    ## Train & Val Datasets 
    max_duration = args.max_duration
    train_dataset = Dataset(
        examples=df_train, 
        feature_extractor=feature_extractor, 
        max_duration=max_duration,
        augmentation=args.augmentation
        )
    val_dataset = Dataset(
        examples=df_val, 
        feature_extractor=feature_extractor, 
        max_duration=max_duration
        )

    ## Training Arguments and Class Weights
    training_arguments = define_training_args(
        output_dir=output_dir, 
        batch_size=args.batch, 
        num_steps=args.steps, 
        lr=args.lr, 
        gradient_accumulation_steps=1,
        warmup_steps=args.warmup_steps,
        )
    #class_weights = compute_class_weights(df_train)  #2个class_weights？？


    print_lora_parameters(model)
    ## Trainer 
    trainer = WeightedTrainer(
        #class_weights=class_weights,
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_binary if args.label=='s/p' else compute_metrics
        )

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%")
    print_trainable_parameters(model)
    ## Train and Evaluate
    trainer.train()
    # trainer.evaluate()

     ## Evaluate
    print("----------------------------------")
    df_test = pd.read_csv(args.df_test, index_col=None)
    severity_map = {"0": 0, "High": 1, "Mid": 2, "Low": 3, "Very Low": 4} 
    if 'severity' in df_test.columns and df_test['severity'].dtype == object:
        df_test['severity'] = df_test['severity'].map(severity_map)
        print("Label2Id: ", label2id)
        print("Id2Label: ", id2label)
    for index in range(0,len(df_test)):
        df_test.loc[index,'label'] = label2id[str(df_test.loc[index,args.label])]
    df_test['label'] = df_test['label'].astype(int)
    test_dataset = Dataset(
        examples=df_test, 
        feature_extractor=feature_extractor, 
        max_duration=max_duration
        )
    
    if local_rank in [-1, 0]:    
        print("Evaluating...")
    predictions = trainer.predict(test_dataset)
    if local_rank in [-1, 0]:
        with open(f"{output_dir}/predictions-test.txt", "w") as f:
            f.write(f"Accuracy: {predictions.metrics['test_accuracy']}\n")
            f.write(f"F1: {predictions.metrics['test_f1_macro']}\n")
            if args.label == "s/p":
                f.write(f"AUC: {predictions.metrics['test_auc']}\n")
            else:
                f.write(f"Precision (M): {predictions.metrics['test_precision_macro']}\n")
                f.write(f"Recall (M): {predictions.metrics['test_recall_macro']}\n")
                f.write(f"Top-3 Accuracy: {predictions.metrics['test_top3_accuracy']}\n")
        print("------------------------------------")
        print("Running with the following parameters:")
        print("Batch size: ", args.batch)
        print("Number of steps: ", args.steps)
        print("Model: ", args.model)
        print("Warmup steps: ", args.warmup_steps)
        print("Learning rate: ", args.lr)
        print("Maximum audio duration: ", args.max_duration)
        print("Label to predict: ", args.label)
        print("------------------------------------\n")

    ## Compute confidence scores
    if args.save_confidence_scores:
        # 【修改 3】：保存 csv 同样存在读写冲突风险，最好也只让 Rank 0 写入
        if local_rank in [-1, 0]:
            print("Saving confidence scores...")
            sof = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
            df = pd.DataFrame(sof, columns=labels)
            df["category"] = [id2label[int(label_id)] for label_id in predictions.label_ids]
            df.to_csv(f"{output_dir}/confidence_scores.csv", index=False)
    
    # dist.destroy_process_group()
    # 安全销毁分布式进程组
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            dist.barrier()  # 等待所有进程同步
            dist.destroy_process_group()
    except Exception as e:
        print(f"销毁进程组时忽略错误: {e}")