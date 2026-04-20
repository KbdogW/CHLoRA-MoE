# CHLoRA-MoE Project

## 网页界面截图

示例：
![Webpage Screenshot 1](./screenshots/1.png)
![Webpage Screenshot 2](./screenshots/2.png)
![Webpage Screenshot 2](./screenshots/3.png)

## 简介
这是一个使用 Triton Inference Server（后端：FastAPI和前端：Vue/Vite ）构建的深度学习推理与 RAG 系统项目。

## 包含的模块
- **frontend**: Vue 3 + Vite 前端界面
- **main.py (Backend)**: 基于 FastAPI 构建的高并发异步后端服务。负责接收前端音频上传，调用 Triton 推理服务，并结合 Qdrant 向量检索与 LLM（大语言模型）生成具有人文关怀的医疗临床评估报告。
- **rag**: 检索增强生成 (RAG) 知识库构建与查询（结合了 BGE Embedding 和 Qdrant）
- **train_model**: 模型微调与训练脚本
- **triton_model_repo**: Triton 推理服务器的模型仓库配置
- **其他脚本**: 用于导出 ONNX、批处理客户端测试、性能数据准备等
