import torch
import librosa
import numpy as np
import copy


""" Dataset Class """
class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, max_duration, augmentation=False):
        self.examples = examples['wav']
        self.labels = examples['label']
        self.severity_labels = examples['severity'] # --- [新增] 处理严重程度标签 ---
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.augmentation = augmentation
        self.sr = 16_000
        # dataset.py
        if 'severity' in examples:
            self.severity_labels = examples['severity']
            print(f"Severity distribution: {np.unique(self.severity_labels, return_counts=True)}") # <--- 检查这个！
    # def __getitem__(self, idx):

    #     ## Augmentation:
    #         # 1: Add noise
    #         # 2: Change speed up
    #         # 3: Change pitch
    #         # 4: Change speed down
    #         # 5: Add noise + Change speed (up) + Change pitch
    #         # 6: Add noise + Change speed (down) + Change pitch
    #     if self.augmentation:
    #         # Augment or not, with a probability of 0.30
    #         # p=[0.15, 0.85] for tts, p=[0.30, 0.70] for everything else
    #         augment = np.random.choice([True, False], p=[0.30, 0.70]) 
    #         # Choose augmentation type
    #         augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
    #         if augment:
    #             try:
    #                 audio, sr = librosa.load(self.examples[idx], sr=self.sr)
    #                 if augmentation_type == 1:
    #                     # Add noise
    #                     noise = np.random.normal(0, 0.005, audio.shape[0])
    #                     audio = audio + noise
    #                 elif augmentation_type == 2:
    #                     # Change speed up
    #                     audio = librosa.effects.time_stretch(audio, rate=1.2)
    #                 elif augmentation_type == 3:
    #                     # Change pitch
    #                     audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
    #                 elif augmentation_type == 4:
    #                     # Change speed down
    #                     audio = librosa.effects.time_stretch(audio, rate=0.8)
    #                 elif augmentation_type == 5:
    #                     # Add noise + Change speed (up) + Change pitch
    #                     noise = np.random.normal(0, 0.005, audio.shape[0])
    #                     audio = audio + noise
    #                     audio = librosa.effects.time_stretch(audio, rate=1.2)
    #                     audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
    #                 elif augmentation_type == 6:
    #                     # Add noise + Change speed (down) + Change pitch
    #                     noise = np.random.normal(0, 0.005, audio.shape[0])
    #                     audio = audio + noise
    #                     audio = librosa.effects.time_stretch(audio, rate=0.8)
    #                     audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
    #                 # Extract features
    #                 inputs = self.feature_extractor(
    #                     audio.squeeze(),
    #                     sampling_rate=self.feature_extractor.sampling_rate, 
    #                     return_tensors="pt",
    #                     max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
    #                     truncation=True,
    #                     padding='max_length')
    #             except:
    #                 print("Audio not available", self.examples[idx])

    #         else:
    #             try:
    #                 inputs = self.feature_extractor(
    #                     librosa.load(self.examples[idx], sr=self.sr)[0].squeeze(),
    #                     sampling_rate=self.feature_extractor.sampling_rate, 
    #                     return_tensors="pt",
    #                     max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
    #                     truncation=True,
    #                     padding='max_length')
    #             except:
    #                 print("Audio not available: ", self.examples[idx])
    #     ## No augmentation
    #     else:
    #         try:
    #             inputs = self.feature_extractor(
    #                 librosa.load(self.examples[idx], sr=self.sr)[0].squeeze(),
    #                 sampling_rate=self.feature_extractor.sampling_rate, 
    #                 return_tensors="pt",
    #                 max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
    #                 truncation=True,
    #                 padding='max_length')
    #         except:
    #             print("Audio not available", self.examples[idx])

    #     try:
    #         item = {'input_values': inputs['input_values'].squeeze(0)}
    #         item["labels"] = torch.tensor(self.labels[idx])
    #         # --- [新增] 返回严重程度标签 ---
    #         item["severity_labels"] = torch.tensor(self.severity_labels[idx])
    #     except:
    #         item = { 'input_values': [], 'labels': [], 'severity_labels': [] }
    #     return item

    def __getitem__(self, idx):

        ## Augmentation:
            # 1: Add noise
            # 2: Change speed up
            # 3: Change pitch
            # 4: Change speed down
            # 5: Add noise + Change speed (up) + Change pitch
            # 6: Add noise + Change speed (down) + Change pitch
        if self.augmentation:
            # Augment or not, with a probability of 0.30
            # p=[0.15, 0.85] for tts, p=[0.30, 0.70] for everything else
            augment = np.random.choice([True, False], p=[0.30, 0.70]) 
            # Choose augmentation type
            augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
            # 1. 先加载音频
            try:
                audio, sr = librosa.load(self.examples[idx], sr=self.sr)
            except:
                print("Audio not available", self.examples[idx])
                # 如果加载失败，给一个全0的空音频防止报错
                audio = np.zeros(int(self.sr * self.max_duration))

            # 2. 执行原有的增强逻辑 (如果选中 augment)
            if augment:
                if augmentation_type == 1:
                    # Add noise
                    noise = np.random.normal(0, 0.005, audio.shape[0])
                    audio = audio + noise
                elif augmentation_type == 2:
                    # Change speed up
                    audio = librosa.effects.time_stretch(audio, rate=1.2)
                elif augmentation_type == 3:
                    # Change pitch
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                elif augmentation_type == 4:
                    # Change speed down
                    audio = librosa.effects.time_stretch(audio, rate=0.8)
                elif augmentation_type == 5:
                    # Add noise + Change speed (up) + Change pitch
                    noise = np.random.normal(0, 0.005, audio.shape[0])
                    audio = audio + noise
                    audio = librosa.effects.time_stretch(audio, rate=1.2)
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                elif augmentation_type == 6:
                    # Add noise + Change speed (down) + Change pitch
                    noise = np.random.normal(0, 0.005, audio.shape[0])
                    audio = audio + noise
                    audio = librosa.effects.time_stretch(audio, rate=0.8)
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)

            # ============================================================
            # [新增] 强力 Time Masking (Cutout)
            # ============================================================
            # 逻辑：独立于上面的 augment，额外有 50% 的概率对音频进行“抹除”
            # 这种双重随机性（Double Randomness）能极大增强泛化能力
            if np.random.random() < 0.5: 
                # 随机决定掩盖的长度：0.05秒 到 0.8秒 之间 (对于病理语音，不要盖太长)
                mask_duration = np.random.uniform(0.05, 0.8)
                mask_len_samples = int(self.sr * mask_duration)
                
                # 确保音频长度足够掩盖
                if audio.shape[0] > mask_len_samples:
                    # 随机选择起始点
                    start_sample = np.random.randint(0, audio.shape[0] - mask_len_samples)
                    # 将该段音频置为静音 (0)
                    audio[start_sample : start_sample + mask_len_samples] = 0.0
            # ============================================================

            # 3. 提取特征 (input_values)
            # 注意：把原来的 inputs 提取代码移到这里，统一处理
            try:
                inputs = self.feature_extractor(
                    audio.squeeze(), # audio 已经是 numpy 数组了
                    sampling_rate=self.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length')
            except Exception as e:
                print(f"Feature extraction failed: {e}")
                # 返回空特征以防崩溃
                inputs = {'input_values': torch.zeros(1, int(self.sr * self.max_duration))}

        ## No augmentation (验证集/测试集逻辑)
        else:
            try:
                # 验证集不做任何掩盖，保持原样
                inputs = self.feature_extractor(
                    librosa.load(self.examples[idx], sr=self.sr)[0].squeeze(),
                    sampling_rate=self.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length')
            except:
                print("Audio not available", self.examples[idx])
                inputs = {'input_values': torch.zeros(1, int(self.sr * self.max_duration))}


        try:
            item = {'input_values': inputs['input_values'].squeeze(0)}
            item["labels"] = torch.tensor(self.labels[idx])
            # --- [新增] 返回严重程度标签 ---
            item["severity_labels"] = torch.tensor(self.severity_labels[idx])
        except:
            print("!not available")
            item = { 'input_values': [], 'labels': [], 'severity_labels': [] }
        return item

    def __len__(self):
        return len(self.examples)