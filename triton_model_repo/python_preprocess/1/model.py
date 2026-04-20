import triton_python_backend_utils as pb_utils
import numpy as np
import io
import librosa
from transformers import Wav2Vec2FeatureExtractor

SAMPLE_RATE = 16000
MAX_DURATION = 3.0

class TritonPythonModel:
    def initialize(self, args):
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # 初始化 Feature Extractor，与 dataset.py 一致
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")

    def execute(self, requests):
        responses = []
        for request in requests:
            # 获取输入张量 (字节流)
            in_tensor = pb_utils.get_input_tensor_by_name(request, "audio_bytes")
            
            # Triton 字符串张量解码
            audio_bytes_array = in_tensor.as_numpy()
            
            input_values_list = []
            
            for item in audio_bytes_array:
                raw_bytes = item[0]
                
                try:
                    # 使用 io.BytesIO 和 librosa 读取内存中的二进制音频，指定采样率与 dataset.py 保持一致
                    with io.BytesIO(raw_bytes) as buf:
                        audio, _ = librosa.load(buf, sr=SAMPLE_RATE)
                    
                    # 完全照搬 dataset.py 中的特征提取逻辑（无 VAD，保持原样提取）
                    inputs = self.feature_extractor(
                        audio.squeeze(),
                        sampling_rate=self.feature_extractor.sampling_rate, 
                        return_tensors="np", # 适配 Triton，这里改用 numpy 而不是 pt
                        max_length=int(self.feature_extractor.sampling_rate * MAX_DURATION), 
                        truncation=True,
                        padding='max_length'
                    )
                    input_values_list.append(inputs['input_values'][0])
                except Exception as e:
                    print(f"Audio processing failed: {e}")
                    # 照搬 dataset.py 中失败时返回全零矩阵的防崩溃逻辑
                    input_values_list.append(np.zeros(int(SAMPLE_RATE * MAX_DURATION), dtype=np.float32))
                
            # 拼接成 Batch，输出类型为 FP32
            out_array = np.stack(input_values_list, axis=0).astype(np.float32)
            
            # 创建输出张量
            out_tensor = pb_utils.Tensor("input_values", out_array)
            
            # 生成 Response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
            
        return responses

    def finalize(self):
        pass
