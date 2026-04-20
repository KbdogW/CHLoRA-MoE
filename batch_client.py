import argparse
import numpy as np
import sys
import os
import glob
import csv
import tritonclient.http as httpclient
from sklearn.metrics import accuracy_score, classification_report

def infer_chunk(triton_client, audio_paths):
    """向 Triton 发送一批音频请求，返回预测的 logit 结果"""
    batch_audio_data = []
    valid_paths = []
    
    # 1. 组装 Batch
    for path in audio_paths:
        try:
            with open(path, "rb") as f:
                batch_audio_data.append([f.read()])
            valid_paths.append(path)
        except Exception as e:
            pass # 忽略读取失败的文件

    if not batch_audio_data:
        return []

    # 2. 构造成 [N, 1] 的 NumPy 数组，dtype 为 object
    input_data = np.array(batch_audio_data, dtype=object)

    inputs = []
    inputs.append(httpclient.InferInput("audio_bytes", input_data.shape, "BYTES"))
    inputs[0].set_data_from_numpy(input_data)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput("logits"))
    outputs.append(httpclient.InferRequestedOutput("router_logits"))

    # 3. 发送请求
    try:
        results = triton_client.infer(
            model_name="ensemble_model",
            inputs=inputs,
            outputs=outputs
        )
    except Exception as e:
        print(f"部分批量推理失败，已跳过: {e}")
        return []

    # 4. 解析结果
    logits = results.as_numpy("logits")
    
    predictions = []
    for i, path in enumerate(valid_paths):
        l_tensor = logits[i]
        probs = np.exp(l_tensor) / np.sum(np.exp(l_tensor))
        pred_class = np.argmax(probs)
        predictions.append(pred_class)
        
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='Directory containing audio files')
    parser.add_argument('-f', '--files', type=str, nargs='+', help='List of audio files')
    parser.add_argument('-c', '--csv', type=str, help='CSV file containing test set (must have "wav" column, optional "severity" column)')
    parser.add_argument('-u', '--url', type=str, default='localhost:9000', help='Inference server URL')
    args = parser.parse_args()
    
    try:
        triton_client = httpclient.InferenceServerClient(url=args.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    audio_data_list = [] # 存放 (path, true_label)
    
    if args.dir:
        paths = glob.glob(os.path.join(args.dir, "*.wav"))
        audio_data_list.extend([(p, None) for p in paths])
    if args.files:
        audio_data_list.extend([(p, None) for p in args.files])
    if args.csv:
        print(f"正在从 {args.csv} 读取测试集...")
        with open(args.csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                wav_path = row.get('wav', '').strip()
                # 解析 severity 标签 (支持文本格式)
                severity = row.get('severity', '').strip()
                severity_map = {"0": 0, "High": 1, "Mid": 2, "Low": 3, "Very Low": 4}
                
                true_label = None
                if severity.isdigit() and 0 <= int(severity) <= 4:
                    true_label = int(severity)
                elif severity in severity_map:
                    true_label = severity_map[severity]
                elif severity.lower() in [k.lower() for k in severity_map.keys()]:
                    # 尝试忽略大小写匹配
                    for k, v in severity_map.items():
                        if severity.lower() == k.lower():
                            true_label = v
                            break
                if wav_path:
                    audio_data_list.append((wav_path, true_label))
                    
    if not audio_data_list:
        print("未找到任何音频文件，请检查参数！")
        sys.exit(1)
        
    total_files = len(audio_data_list)
    print(f"准备测试总音频数量: {total_files}")
    
    # 按照 Triton 配置的 max_batch_size (32) 进行分块处理
    BATCH_SIZE = 32
    
    all_true_labels = []
    all_pred_labels = []
    
    # 进度提示
    import time
    start_time = time.time()
    
    for i in range(0, total_files, BATCH_SIZE):
        batch_items = audio_data_list[i:i+BATCH_SIZE]
        batch_paths = [item[0] for item in batch_items]
        batch_trues = [item[1] for item in batch_items]
        
        # 调用服务端进行并行推理
        preds = infer_chunk(triton_client, batch_paths)
        
        if len(preds) == len(batch_trues):
            for j in range(len(preds)):
                if batch_trues[j] is not None:
                    all_true_labels.append(batch_trues[j])
                    all_pred_labels.append(preds[j])
                    
        # 打印进度条
        progress = min(total_files, i + BATCH_SIZE)
        sys.stdout.write(f"\r处理进度: {progress}/{total_files} [{progress/total_files*100:.1f}%]")
        sys.stdout.flush()

    end_time = time.time()
    
    print(f"\n\n========================================")
    print(f"测试完成！总耗时: {end_time - start_time:.2f} 秒")
    print(f"吞吐量 (QPS): {total_files / (end_time - start_time):.2f} audio/s")
    print(f"========================================")

    # 如果 CSV 文件里有标签，顺便计算一下准确率
    if all_true_labels:
        acc = accuracy_score(all_true_labels, all_pred_labels)
        print(f"测试集总准确率 (Accuracy): {acc * 100:.2f}%")
        print("\n详细分类报告:")
        print(classification_report(all_true_labels, all_pred_labels))
