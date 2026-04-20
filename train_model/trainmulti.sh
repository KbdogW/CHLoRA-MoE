#!/bin/bash

# 使用方法: bash train_multi.sh <显卡列表> <种子>
# 例如: bash train_multi.sh 0,1,2,3 42  (使用4张卡)
# 例如: bash train_multi.sh 4,5 42      (使用2张卡)

gpu_list=$1
seed=$2

# 1. 检查参数
if [ -z "$gpu_list" ] || [ -z "$seed" ]; then
    echo "❌ 错误: 参数缺失！"
    echo "💡 使用方法: bash train_multi.sh <显卡列表> <种子>"
    echo "   例如(使用4张卡): bash train_multi.sh 0,1,2,3 42"
    echo "   例如(使用2张卡): bash train_multi.sh 0,1 42"
    exit 1
fi

# 2. 计算显卡数量 (通过逗号分隔符计算)
# 如果输入是 0,1,2,3 -> 数量是 4
# 如果输入是 4,5 -> 数量是 2
gpu_count=$(echo $gpu_list | tr ',' '\n' | wc -l)

echo "========================================"
echo "🚀 开始多卡分布式训练 (DDP)"
echo "指定显卡 (Physical GPUs): $gpu_list"
echo "显卡数量 (nproc): $gpu_count"
echo "随机种子 (Seed): $seed"
echo "========================================"

# 3. 设置环境变量并启动
# OMP_NUM_THREADS=1 防止CPU过载
# CUDA_VISIBLE_DEVICES 限制程序只能看到你指定的显卡

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$gpu_list

# 使用 python3 -m torch.distributed.run 代替 torchrun 防止找不到命令
# master_port 随机设一个，防止端口冲突
# facebook/hubert-large-ls960-ft

python3 -m torch.distributed.run \
    --nproc_per_node=$gpu_count \
    --master_port=29400 \
    finetuning.py \
    --df_train IDtrain.csv \
    --df_val IDtest.csv \
    --feature_extractor microsoft/wavlm-large \
    --model microsoft/wavlm-large \
    --output_dir "output/" \
    --label severity \
    --lr 5e-5 \
    --steps 18000 \
    --warmup_steps 2000 \
    --df_test IDtest.csv \
    --augmentation \
    --batch 32 \
    --seed $seed 
    # --save_confidence_scores
echo "✅ Finished training with seed: $seed on GPU: $gpu_list"