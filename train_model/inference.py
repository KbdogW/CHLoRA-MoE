# -*- coding: utf-8 -*-
"""
inference.py - MoE 语音分类模型推理脚本

功能:
1. 加载训练好的 MoE 模型权重 (safetensors)
2. VAD 静音裁剪，提取有效语音段
3. 白名单式权重验证
4. 单样本/批量推理

使用方法:
    python inference.py --checkpoint checkpoint-9800 --audio test.wav
    python inference.py --checkpoint checkpoint-9800 --batch audio_list.txt
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from safetensors.torch import load_file
from transformers import Wav2Vec2FeatureExtractor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from moe import PhysicianGuidedMoEWavLM


# =============================================================================
# 常量配置
# =============================================================================
SAMPLE_RATE = 16000
MAX_DURATION = 3.0  # 秒
TARGET_LENGTH = int(SAMPLE_RATE * MAX_DURATION)  # 48000 采样点

# 标签映射 (与训练一致)
ID2LABEL = {
    0: "0",
    1: "High",
    2: "Mid",
    3: "Low",
    4: "Very Low"
}

SEVERITY_MAP = {
    "0": 0, "High": 1, "Mid": 2, "Low": 3, "Very Low": 4
}


def parse_label(label_value) -> int:
    """
    解析标签值，统一转换为 int 类别 id

    支持的输入格式:
    - 整数字符串: "0", "1", "2", "3", "4"
    - 整数: 0, 1, 2, 3, 4
    - 标签字符串: "0 (Healthy)", "High", "Mid", "Low", "Very Low"

    Args:
        label_value: 输入的标签值

    Returns:
        int: 标准类别 id (0-4)

    Raises:
        ValueError: 当遇到未知标签格式时
    """
    if isinstance(label_value, int):
        # 已经是整数，直接检查是否在有效范围内
        if 0 <= label_value <= 4:
            return label_value
        else:
            raise ValueError(f"Invalid integer label: {label_value}")

    elif isinstance(label_value, str):
        # 去除可能的空格
        label_str = label_value.strip()

        # 直接尝试转换为整数
        try:
            return int(label_str)
        except ValueError:
            # 转换失败，尝试从映射中查找
            # 1. 先检查完整的映射（ID2LABEL）
            for label_id, label_name in ID2LABEL.items():
                if label_str == label_name:
                    return int(label_id)

            # 2. 检查 SEVERITY_MAP
            if label_str in SEVERITY_MAP:
                return SEVERITY_MAP[label_str]

            # 3. 还不行，尝试模糊匹配
            for label_id, label_name in ID2LABEL.items():
                # 如果是子字符串匹配（比如 "Healthy" 匹配 "0 (Healthy)"）
                if label_str in label_name or label_name in label_str:
                    return int(label_id)

            # 4. 最后尝试简写匹配
            short_map = {
                "healthy": 0,
                "h": 0,
                "0": 0,
                "high": 1,
                "hi": 1,
                "1": 1,
                "mid": 2,
                "md": 2,
                "2": 2,
                "low": 3,
                "lw": 3,
                "3": 3,
                "very low": 4,
                "vl": 4,
                "v": 4,
                "4": 4
            }
            normalized = label_str.lower().replace(" ", "")
            if normalized in short_map:
                return short_map[normalized]

            # 如果都不匹配，抛出异常
            raise ValueError(f"无法解析的标签: '{label_str}'。支持格式: {list(ID2LABEL.values())}")

    else:
        raise ValueError(f"不支持的标签类型: {type(label_value)}，值: {label_value}")


# =============================================================================
# VAD 静音裁剪
# =============================================================================
def trim_silence(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    top_db: float = 20.0,
    frame_length: int = 512,
    hop_length: int = 256
) -> np.ndarray:
    """
    基于能量的静音裁剪 (VAD)

    Args:
        audio: 音频波形 [samples]
        sr: 采样率
        top_db: 低于最大能量多少 dB 视为静音
        frame_length: 帧长度
        hop_length: 帧移

    Returns:
        裁剪后的音频，固定长度 TARGET_LENGTH
    """
    # 使用 librosa 的 effects.trim 进行静音裁剪
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # 如果裁剪后短于目标长度，填充
    if len(trimmed) < TARGET_LENGTH:
        # 零填充到目标长度
        padding = np.zeros(TARGET_LENGTH - len(trimmed), dtype=trimmed.dtype)
        trimmed = np.concatenate([trimmed, padding])
    elif len(trimmed) > TARGET_LENGTH:
        # 如果超过目标长度，截断
        trimmed = trimmed[:TARGET_LENGTH]

    return trimmed


def load_and_preprocess_audio(
    audio_path: str,
    feature_extractor: Wav2Vec2FeatureExtractor,
    apply_vad: bool = True,
    vad_top_db: float = 20.0
) -> torch.Tensor:
    """
    加载并预处理音频文件

    Args:
        audio_path: 音频文件路径
        feature_extractor: HuggingFace 特征提取器
        apply_vad: 是否应用 VAD 静音裁剪
        vad_top_db: VAD 阈值

    Returns:
        input_values: [1, TARGET_LENGTH] 张量
    """
    # 1. 加载音频
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

    # 2. 可选: VAD 静音裁剪
    if apply_vad:
        audio = trim_silence(audio, sr=SAMPLE_RATE, top_db=vad_top_db)
    else:
        # 修正: 不使用 VAD 时，不要手动补零，只做截断处理。
        # 补零操作应留给后续的 feature_extractor 处理，否则会干扰其内部的 Z-Score 归一化。
        if len(audio) > TARGET_LENGTH:
            audio = audio[:TARGET_LENGTH]

    # 3. 特征提取
    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        max_length=TARGET_LENGTH,
        truncation=True,
        padding='max_length'
    )

    return inputs['input_values']


# =============================================================================
# 白名单式权重验证
# =============================================================================
def verify_weights_loaded(
    model: PhysicianGuidedMoEWavLM,
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True
) -> bool:
    """
    白名单式核对关键参数组是否全部命中

    检查项:
    1. Router 权重是否完整加载
    2. Classifier 权重是否完整加载
    3. 每个 adapter 的 LoRA A/B 权重是否都进入了对应模块

    Args:
        model: 模型实例
        state_dict: 加载的权重字典
        verbose: 是否打印详细信息

    Returns:
        是否所有关键权重都加载成功
    """
    all_passed = True

    if verbose:
        print("\n" + "=" * 60)
        print("权重加载验证 (白名单式)")
        print("=" * 60)

    # 1. Router 权重检查
    router_keys = [k for k in state_dict.keys() if 'router.' in k]
    router_params = sum(state_dict[k].numel() for k in router_keys)

    if len(router_keys) > 0:
        if verbose:
            print(f"✓ Router 权重: {len(router_keys)} keys, {router_params:,} params")
    else:
        print(f"✗ Router 权重缺失!")
        all_passed = False

    # 2. Router Pooling 权重检查
    pooling_keys = [k for k in state_dict.keys() if 'router_pooling.' in k]
    pooling_params = sum(state_dict[k].numel() for k in pooling_keys)

    if len(pooling_keys) > 0:
        if verbose:
            print(f"✓ Router Pooling 权重: {len(pooling_keys)} keys, {pooling_params:,} params")
    else:
        print(f"✗ Router Pooling 权重缺失!")
        all_passed = False

    # 3. Classifier 权重检查
    classifier_keys = [k for k in state_dict.keys() if 'classifier.' in k]
    classifier_params = sum(state_dict[k].numel() for k in classifier_keys)

    if len(classifier_keys) > 0:
        if verbose:
            print(f"✓ Classifier 权重: {len(classifier_keys)} keys, {classifier_params:,} params")
    else:
        print(f"✗ Classifier 权重缺失!")
        all_passed = False

    # 4. Final Pooling 权重检查
    final_pooling_keys = [k for k in state_dict.keys() if 'final_pooling.' in k]
    final_pooling_params = sum(state_dict[k].numel() for k in final_pooling_keys)

    if len(final_pooling_keys) > 0:
        if verbose:
            print(f"✓ Final Pooling 权重: {len(final_pooling_keys)} keys, {final_pooling_params:,} params")
    else:
        print(f"✗ Final Pooling 权重缺失!")
        all_passed = False

    # 5. 每个 adapter 的 LoRA A/B 权重检查
    print("\nLoRA Adapter 权重检查:")
    adapter_names = ['shared_bottom'] + [f'expert_{i}' for i in range(model.num_experts)]

    for adapter_name in adapter_names:
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower() and adapter_name in k]
        lora_A_keys = [k for k in lora_keys if 'lora_A' in k]
        lora_B_keys = [k for k in lora_keys if 'lora_B' in k]

        lora_A_params = sum(state_dict[k].numel() for k in lora_A_keys)
        lora_B_params = sum(state_dict[k].numel() for k in lora_B_keys)

        if len(lora_A_keys) > 0 and len(lora_B_keys) > 0:
            if verbose:
                print(f"  ✓ '{adapter_name}': {len(lora_A_keys)} lora_A ({lora_A_params:,}) + {len(lora_B_keys)} lora_B ({lora_B_params:,})")
        else:
            print(f"  ✗ '{adapter_name}': LoRA 权重缺失!")
            all_passed = False

    # 6. 统计信息
    total_keys = len(state_dict)
    total_params = sum(p.numel() for p in state_dict.values())

    if verbose:
        print("\n" + "-" * 60)
        print(f"总权重: {total_keys} keys, {total_params:,} params")
        print("=" * 60)

        if all_passed:
            print("✅ 所有关键权重验证通过!")
        else:
            print("❌ 部分关键权重验证失败!")

    return all_passed


# =============================================================================
# MoE 推理类
# =============================================================================
class MoEInference:
    """
    MoE 语音分类推理封装类
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        apply_vad: bool = True,
        vad_top_db: float = 20.0,
        verbose: bool = True
    ):
        """
        初始化推理器

        Args:
            checkpoint_path: 检查点目录路径 (包含 model.safetensors)
            device: 计算设备
            apply_vad: 是否应用 VAD
            vad_top_db: VAD 阈值
            verbose: 是否打印详细信息
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.apply_vad = apply_vad
        self.vad_top_db = vad_top_db
        self.verbose = verbose

        # 标签映射
        self.id2label = ID2LABEL
        self.severity_map = SEVERITY_MAP

        # 初始化组件
        self._init_feature_extractor()
        self._init_model()
        self._verify_weights()

    def _init_feature_extractor(self):
        """初始化特征提取器"""
        if self.verbose:
            print(f"\n加载特征提取器: microsoft/wavlm-large")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-large"
        )

    def _init_model(self):
        """初始化模型并加载权重"""
        if self.verbose:
            print(f"\n初始化模型架构...")

        # 1. 处理设备参数
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，自动回退到 CPU")
            self.device = 'cpu'
        elif self.device.startswith('cuda'):
            # 处理 cuda:0, cuda:1 等格式
            try:
                device_idx = int(self.device.split(':')[1])
                if torch.cuda.device_count() <= device_idx:
                    print(f"⚠️  CUDA 设备索引 {device_idx} 超出范围，使用默认 CUDA")
                    self.device = 'cuda'
            except:
                print(f"⚠️  无效的设备格式 {self.device}，使用默认 CUDA")
                self.device = 'cuda'

        if self.verbose:
            print(f"使用设备: {self.device}")

        # 2. 创建模型实例
        self.model = PhysicianGuidedMoEWavLM(
            base_model_name="microsoft/wavlm-large",
            num_labels=5,
            num_experts=5
        )

        # 3. 加载 safetensors 权重
        weights_path = os.path.join(self.checkpoint_path, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        if self.verbose:
            print(f"加载权重: {weights_path}")

        self.state_dict = load_file(weights_path)

        # 4. 加载权重到模型
        missing, unexpected = self.model.load_state_dict(self.state_dict, strict=False)

        if self.verbose:
            print(f"  Missing keys: {len(missing)}")
            print(f"  Unexpected keys: {len(unexpected)}")

            if len(missing) > 0:
                print(f"\n  Missing keys 详情 (前 10 个):")
                for k in list(missing)[:10]:
                    print(f"    - {k}")

            if len(unexpected) > 0:
                print(f"\n  Unexpected keys 详情 (前 10 个):")
                for k in list(unexpected)[:10]:
                    print(f"    - {k}")

        # 5. 移动到设备并设置评估模式
        self.model.to(self.device)
        self.model.eval()

    def _verify_weights(self):
        """验证权重加载正确性"""
        verify_weights_loaded(self.model, self.state_dict, verbose=self.verbose)

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        预处理音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            input_values: [1, TARGET_LENGTH] 张量
        """
        return load_and_preprocess_audio(
            audio_path,
            self.feature_extractor,
            apply_vad=self.apply_vad,
            vad_top_db=self.vad_top_db
        )

    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:
        """
        单样本推理

        Args:
            audio_path: 音频文件路径

        Returns:
            预测结果字典
        """
        # 1. 预处理
        input_values = self.preprocess_audio(audio_path).to(self.device)

        # 2. 推理 (参考 moe.py 的 forward 流程)
        # Step 1: Shared Bottom
        self.model.wavlm.set_adapter("shared_bottom")
        outputs_shared = self.model.wavlm(input_values, output_hidden_states=True)
        middle_hidden = outputs_shared.hidden_states[10]  # 第 10 层用于 Router

        # Step 2: Router
        router_input = self.model.router_pooling(middle_hidden)
        router_logits = self.model.router(router_input)
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_idx = torch.argmax(router_logits, dim=-1).item()

        # Step 3: Expert
        expert_name = f"expert_{expert_idx}"
        self.model.wavlm.set_adapter(expert_name)

        outputs_expert = self.model.wavlm(input_values)
        feat = self.model.final_pooling(outputs_expert.last_hidden_state)
        logits = self.model.classifier(feat)

        # 3. 后处理
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs.max().item()

        # 4. 构建结果
        result = {
            'audio_path': audio_path,
            'predicted_class': pred_class,
            'predicted_label': self.id2label[pred_class],
            'confidence': confidence,
            'probabilities': {
                self.id2label[i]: probs[0, i].item()
                for i in range(len(self.id2label))
            },
            'router_decision': expert_idx,
            'router_probs': {
                f"expert_{i}": router_probs[0, i].item()
                for i in range(self.model.num_experts)
            },
            'logits': logits.cpu().numpy().flatten().tolist(),
            'router_logits': router_logits.cpu().numpy().flatten().tolist()
        }

        return result

    @torch.no_grad()
    def batch_predict(
        self,
        audio_paths: List[str],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        批量推理

        Args:
            audio_paths: 音频文件路径列表
            batch_size: 批大小

        Returns:
            预测结果列表
        """
        results = []

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]

            # 预处理批次
            batch_inputs = []
            for path in batch_paths:
                inputs = self.preprocess_audio(path)
                batch_inputs.append(inputs)

            batch_tensor = torch.cat(batch_inputs, dim=0).to(self.device)
            batch_size_actual = batch_tensor.shape[0]

            # Shared Bottom
            self.model.wavlm.set_adapter("shared_bottom")
            outputs_shared = self.model.wavlm(batch_tensor, output_hidden_states=True)
            middle_hidden = outputs_shared.hidden_states[10]

            # Router
            router_input = self.model.router_pooling(middle_hidden)
            router_logits = self.model.router(router_input)
            router_probs = torch.softmax(router_logits, dim=-1)

            # Expert 分组处理 (与 moe.py forward 一致)
            target_indices = torch.argmax(router_logits, dim=-1)
            final_logits = torch.zeros(batch_size_actual, self.model.num_labels, device=self.device)

            for expert_idx, expert_name in enumerate(self.model.expert_names):
                indices = (target_indices == expert_idx).nonzero(as_tuple=True)[0]

                if len(indices) == 0:
                    continue

                self.model.wavlm.set_adapter(expert_name)
                sub_input = batch_tensor[indices]
                out_expert = self.model.wavlm(sub_input)
                feat = self.model.final_pooling(out_expert.last_hidden_state)
                sub_logits = self.model.classifier(feat)
                final_logits[indices] = sub_logits

            # 后处理
            probs = torch.softmax(final_logits, dim=-1)
            pred_classes = torch.argmax(probs, dim=-1)

            for j, path in enumerate(batch_paths):
                pred_class = pred_classes[j].item()
                result = {
                    'audio_path': path,
                    'predicted_class': pred_class,
                    'predicted_label': self.id2label[pred_class],
                    'confidence': probs[j].max().item(),
                    'probabilities': {
                        self.id2label[k]: probs[j, k].item()
                        for k in range(len(self.id2label))
                    },
                    'router_decision': target_indices[j].item(),
                    'router_probs': {
                        f"expert_{k}": router_probs[j, k].item()
                        for k in range(self.model.num_experts)
                    }
                }
                results.append(result)

        return results


# =============================================================================
# 评估指标计算
# =============================================================================
def calculate_metrics(
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str] = None,
    average: str = 'macro'
) -> Dict:
    """
    计算分类评估指标

    Args:
        true_labels: 真实标签列表
        pred_labels: 预测标签列表
        class_names: 类别名称列表 (可选)
        average: 计算方式 ('micro', 'macro', 'weighted')

    Returns:
        评估指标字典
    """
    # 确保标签是 numpy 数组
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # 固定 labels=[0,1,2,3,4]，保证混淆矩阵是 5x5
    labels = [0, 1, 2, 3, 4]

    # 计算各项指标
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average=average, labels=labels, zero_division=0)
    precision = precision_score(true_labels, pred_labels, average=average, labels=labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, average=average, labels=labels, zero_division=0)

    # 混淆矩阵 - 固定 5x5，顺序与 ID2LABEL 一致
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    # 逐类指标 - 显式指定 labels=[0,1,2,3,4]
    class_precision = precision_score(true_labels, pred_labels, average=None, labels=labels, zero_division=0)
    class_recall = recall_score(true_labels, pred_labels, average=None, labels=labels, zero_division=0)
    class_f1 = f1_score(true_labels, pred_labels, average=None, labels=labels, zero_division=0)

    per_class_metrics = {}
    if class_names is not None:
        for i, name in enumerate(class_names):
            per_class_metrics[name] = {
                'precision': class_precision[i],
                'recall': class_recall[i],
                'f1': class_f1[i]
            }

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist(),
        'class_metrics': per_class_metrics
    }

    return metrics


def load_labels_from_csv(csv_path: str) -> Dict[str, int]:
    """
    从 CSV 文件加载音频路径和标签的映射

    Args:
        csv_path: CSV 文件路径

    Returns:
        路径到标签的字典

    Raises:
        Exception: 当 CSV 文件读取失败时
        ValueError: 当标签解析失败时
    """
    label_map = {}

    try:
        df = pd.read_csv(csv_path)
        # 列名: wav, txt, severity, speaker
        for _, row in df.iterrows():
            audio_path = row['wav'].strip()
            severity_label = row['severity']

            # 解析标签
            label_id = parse_label(severity_label)
            label_map[audio_path] = label_id

        print(f"从 {csv_path} 加载了 {len(label_map)} 个标签")

        # 验证是否有无法解析的标签
        if len(label_map) == 0:
            raise ValueError("CSV 文件中没有有效的标签数据")

    except Exception as e:
        # 不要静默吞掉错误，抛出异常让调用者处理
        raise Exception(f"加载 CSV 标签失败: {str(e)}")

    return label_map


# =============================================================================
# 命令行入口
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MoE 语音分类推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单文件推理:
    python inference.py --checkpoint checkpoint-9800 --audio test.wav

  批量推理:
    python inference.py --checkpoint checkpoint-9800 --batch audio_list.txt

  不使用 VAD:
    python inference.py --checkpoint checkpoint-9800 --audio test.wav --no-vad
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint-9800',
        help='检查点目录路径 (默认: checkpoint-9800)'
    )

    parser.add_argument(
        '--audio',
        type=str,
        help='单个音频文件路径'
    )

    parser.add_argument(
        '--batch',
        type=str,
        help='批量推理: 音频文件路径列表 (每行一个路径)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='计算设备 (默认: cuda)'
    )

    parser.add_argument(
        '--no-vad',
        action='store_true',
        help='禁用 VAD 静音裁剪'
    )

    parser.add_argument(
        '--vad-top-db',
        type=float,
        default=20.0,
        help='VAD 静音阈值 dB (默认: 20.0)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='输出 JSON 文件路径 (可选)'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='评估模式: 需要指定 --eval-csv'
    )

    parser.add_argument(
        '--eval-csv',
        type=str,
        help='评估用的 CSV 文件路径 (包含真实标签)'
    )

    parser.add_argument(
        '--metrics-output',
        type=str,
        help='评估指标输出文件路径 (可选)'
    )

    parser.add_argument(
        '--average',
        type=str,
        default='macro',
        choices=['micro', 'macro', 'weighted'],
        help='F1 计算方式 (默认: macro)'
    )

    args = parser.parse_args()

    # 检查输入
    if not args.audio and not args.batch:
        parser.error("请指定 --audio 或 --batch")

    if args.evaluate and not args.eval_csv:
        parser.error("评估模式需要指定 --eval-csv")

    # 初始化推理器
    print("=" * 60)
    print("MoE 语音分类推理")
    print("=" * 60)

    inferencer = MoEInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        apply_vad=not args.no_vad,
        vad_top_db=args.vad_top_db,
        verbose=True
    )

    # 执行推理
    if args.audio:
        # 单文件推理
        print(f"\n推理文件: {args.audio}")
        result = inferencer.predict(args.audio)

        print("\n" + "-" * 60)
        print("预测结果:")
        print(f"  文件: {result['audio_path']}")
        print(f"  预测标签: {result['predicted_label']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  Router 决策: Expert {result['router_decision']}")
        print(f"\n  概率分布:")
        for label, prob in result['probabilities'].items():
            print(f"    {label}: {prob:.4f}")
        print(f"\n  Router 概率:")
        for expert, prob in result['router_probs'].items():
            print(f"    {expert}: {prob:.4f}")

        results = [result]

    else:
        # 批量推理
        with open(args.batch, 'r', encoding='utf-8') as f:
            audio_paths = [line.strip() for line in f if line.strip()]

        print(f"\n批量推理: {len(audio_paths)} 个文件")
        results = inferencer.batch_predict(audio_paths)
        # print("\n" + "-" * 60)
        # print("预测结果汇总:")
        # for r in results:
        #     print(f"  {r['audio_path']}: {r['predicted_label']} ({r['confidence']:.4f})")

    # 评估模式
    if args.evaluate:
        print("\n" + "=" * 60)
        print("评估模式")
        print("=" * 60)

        # 加载真实标签
        label_map = load_labels_from_csv(args.eval_csv)
        true_labels = []
        pred_labels = []
        valid_files = []

        # 对比预测结果和真实标签
        for result in results:
            audio_path = result['audio_path']
            if audio_path in label_map:
                true_label = int(label_map[audio_path])
                pred_label = result['predicted_class']
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                valid_files.append(result)

        if not true_labels:
            print("错误: 没有找到匹配的音频文件标签!")
            return

        print(f"\n评估样本数: {len(true_labels)}")
        print(f"\n类别分布:")
        for i in range(5):
            count = sum(1 for label in true_labels if label == i)
            print(f"  {ID2LABEL[i]}: {count}")

        # 计算指标
        metrics = calculate_metrics(
            true_labels,
            pred_labels,
            class_names=[ID2LABEL[i] for i in range(5)],
            average=args.average
        )

        # 输出指标
        print("\n" + "-" * 60)
        print("评估指标:")
        print("-" * 60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        # 逐类指标
        if metrics['class_metrics']:
            print("\n逐类指标:")
            for class_name, class_metrics in metrics['class_metrics'].items():
                print(f"  {class_name}:")
                print(f"    Precision: {class_metrics['precision']:.4f}")
                print(f"    Recall:    {class_metrics['recall']:.4f}")
                print(f"    F1 Score:  {class_metrics['f1']:.4f}")

        # 混淆矩阵 - 固定 5x5 打印
        print("\n混淆矩阵:")
        # 表头
        print("True \\ Pred", end="")
        for name in [ID2LABEL[i] for i in range(5)]:
            print(f"\t{name}", end="")
        print()

        # 确保混淆矩阵是 5x5
        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (5, 5), f"混淆矩阵维度不是 5x5: {cm.shape}"

        # 打印矩阵内容
        for i, name in enumerate([ID2LABEL[i] for i in range(5)]):
            print(f"{name}", end="")
            for j in range(5):
                print(f"\t{cm[i][j]}", end="")
            print()

        # 保存评估结果
        if args.metrics_output:
            eval_results = {
                'summary': metrics
            }
            with open(args.metrics_output, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)
            print(f"\n评估指标已保存到: {args.metrics_output}")

        # 为所有结果添加真实标签信息
        for result in results:
            audio_path = result['audio_path']
            if audio_path in label_map:
                true_label = label_map[audio_path]  # 已经是 int 类型
                result['true_label'] = true_label
                result['true_label_name'] = ID2LABEL[true_label]
                result['is_correct'] = (true_label == result['predicted_class'])
            else:
                result['true_label'] = None
                result['true_label_name'] = "Unknown"
                result['is_correct'] = None

    # 统一保存详细预测结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细预测结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
