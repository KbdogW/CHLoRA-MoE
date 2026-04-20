import struct
import json
import os
import glob

def prepare_data():
    os.makedirs('perf_data', exist_ok=True)
    
    audio_file = 'audio/noisereduce/CF02/CF02_B1_C1_M2.wav'
    if not os.path.exists(audio_file):
        wavs = glob.glob("audio/**/*.wav", recursive=True)
        if wavs: audio_file = wavs[0]

    with open(audio_file, 'rb') as f:
        audio_data = f.read()

    # Triton 的 perf_analyzer 在 is_binary:true 模式下，
    # 依然严格要求 BYTES 类型的二进制流必须带 4 字节的长度前缀！
    # 否则它会把 WAV 的 "RIFF" 头当成长度去解析，导致服务器收到破损的音频流。
    bin_file = 'perf_data/audio_raw.bin'
    with open(bin_file, 'wb') as f:
        f.write(struct.pack('<I', len(audio_data)))
        f.write(audio_data)

    data = {
        "data": [
            {
                "audio_bytes": {
                    "content": ["audio_raw.bin"],
                    "is_binary": True
                }
            }
        ]
    }

    with open('perf_data/data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)
        
    print(f"压测数据已生成！音频真实大小: {len(audio_data)} bytes")

if __name__ == "__main__":
    prepare_data()
