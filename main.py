# -*- coding: utf-8 -*-
"""
main.py - 语音病理评估模型的高并发异步后端服务 (RAG 增强版)

功能:
1. 基于 FastAPI 构建的高性能 RESTful API
2. 提供 `/predict` 接口接收上传的 `.wav` 文件
3. 使用异步非阻塞 I/O (aiofiles) 读取音频文件
4. 集成 Triton Inference Server 异步调用，实现高吞吐
5. 【新增】集成 Qdrant + BGE Embedding 实现本地医疗文献检索 (RAG)
6. 【新增】调用 LLM (兼容 OpenAI 接口) 生成充满人文关怀的临床建议报告
"""

import os
import uuid
import time
import numpy as np
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import tritonclient.http.aio as httpclient

# RAG & LLM 依赖
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import AsyncOpenAI

# =============================================================================
# 全局配置与状态字典
# =============================================================================
TEMP_AUDIO_DIR = "./temp_uploads"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:9000")

# --- RAG 配置 ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "speech_rehab_kb"

# --- LLM 配置 (强烈建议使用环境变量注入) ---
# 这里以兼容 OpenAI SDK 的国内大模型 (如：DeepSeek, 智谱 GLM, 零一万物等) 为例
# 如果你没有这些 Key，可以先去对应的官网申请一个免费额度的 API Key
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-af1ca7541a02482a9258b8d5a5a30877")  # 替换成你的大模型 API Key
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") # 替换成对应的接口地址
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "qwen-flash") # 替换成模型名字

# 标签与诊断说明映射
ID2LABEL = {
    0: "Healthy",
    1: "High",
    2: "Mid",
    3: "Low",
    4: "Very Low"
}

DIAGNOSIS_MAP = {
    0: "未检测到明显的病理语音特征。发音清晰，言语可懂度正常。",
    1: "检测到轻微病理语音特征（言语可懂度高）。可能存在轻微的发声疲劳或早期病变，建议关注嗓音休息并定期观察。",
    2: "检测到中度病理语音特征（言语可懂度中等）。主要表现为发声不稳、部分发音不清等，建议进一步专业评估。",
    3: "检测到重度病理语音特征（言语可懂度低）。存在明显的构音障碍或发声困难，建议尽快就医进行全面检查和干预。",
    4: "检测到极重度病理语音特征（言语可懂度极低）。发音难以辨认，可能伴有严重的声带病变或神经性构音障碍，建议立即进行专业医疗干预。"
}

# 存放全局单例
global_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 启动时初始化：
    1. Triton Client
    2. BGE 向量模型 & Qdrant 检索器
    3. LLM 客户端
    """
    print(f"\n>>> [App Startup] 初始化全局 Triton Client 连接到 {TRITON_SERVER_URL} ...")
    triton_client = httpclient.InferenceServerClient(
        url=TRITON_SERVER_URL,
        conn_limit=100,
        conn_timeout=60.0
    )
    global_state["triton_client"] = triton_client
    
    print("\n>>> [App Startup] 初始化 RAG 检索器 (BGE Model & Qdrant) ...")
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=60.0)
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        global_state["vector_store"] = vector_store
        print(">>> RAG 检索器初始化成功！")
    except Exception as e:
        print(f">>> [警告] RAG 检索器初始化失败: {e}")
        global_state["vector_store"] = None
        
    print("\n>>> [App Startup] 初始化 LLM 客户端 ...")
    llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    global_state["llm_client"] = llm_client
    
    yield
    
    print("\n>>> [App Shutdown] 清理全局连接...")
    await global_state["triton_client"].close()

app = FastAPI(
    title="Severity Assessment API with RAG",
    description="基于 WavLM MoE 的高并发语音自动评估 + RAG 报告生成服务",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cleanup_temp_file(filepath: str):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        pass

# =============================================================================
# 核心接口：/predict
# =============================================================================
@app.post("/predict", summary="上传音频获取评估与康复报告")
async def predict_severity(
    request: Request,
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(..., description="16kHz .wav 格式的语音音频")
):
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="只支持 .wav 格式的音频文件")

    file_id = str(uuid.uuid4())
    temp_filepath = os.path.join(TEMP_AUDIO_DIR, f"{file_id}_{file.filename}")

    try:
        async with aiofiles.open(temp_filepath, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存文件失败: {str(e)}")
    finally:
        await file.close()

    background_tasks.add_task(cleanup_temp_file, temp_filepath)

    start_time = time.time()
    
    # ---------------------------------------------------------
    # 阶段 1: Triton 声学模型推理
    # ---------------------------------------------------------
    try:
        triton_client = global_state["triton_client"]
        async with aiofiles.open(temp_filepath, 'rb') as f:
            audio_data = await f.read()

        input_data = np.array([[audio_data]], dtype=object)
        inputs = [httpclient.InferInput("audio_bytes", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = [
            httpclient.InferRequestedOutput("logits"),
            httpclient.InferRequestedOutput("router_logits")
        ]

        results = await triton_client.infer(
            model_name="ensemble_model",
            inputs=inputs,
            outputs=outputs
        )
        
        logits = results.as_numpy("logits")[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        router_tensor = results.as_numpy("router_logits")[0]
        router_probs = np.exp(router_tensor) / np.sum(np.exp(router_tensor))
        expert_decision = int(np.argmax(router_probs))
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Triton 模型推理失败: {str(e)}")

    inference_time = time.time() - start_time

    severity_label = ID2LABEL.get(pred_class, "Unknown")
    diagnosis_text = DIAGNOSIS_MAP.get(pred_class, "未知状态")

    # ---------------------------------------------------------
    # 阶段 2: RAG 知识检索与 LLM 生成临床建议
    # ---------------------------------------------------------
    rag_report = "知识库报告生成已跳过。"
    rag_time = 0
    
    if global_state.get("vector_store") and global_state.get("llm_client"):
        rag_start = time.time()
        try:
            vector_store = global_state["vector_store"]
            llm_client = global_state["llm_client"]
            
            # 1. 构造搜索 Query (融入 MoE 专家决策)
            query = f"患者被诊断为 {severity_label} 级别的发音障碍，主要声学特征由 Expert_{expert_decision} 捕捉。请提供针对性的言语康复训练和护嗓建议。"
            
            # 2. 向量检索 (Top-3)
            retrieved_docs = vector_store.similarity_search(query, k=3)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 3. 构建 Prompt
            prompt = f"""你是一位专业的嗓音和言语康复理疗师。请根据以下患者的声学AI分析结果和提供的医学参考文献，为患者生成一份充满人文关怀的康复指导报告。

【AI 评估结果】
- 严重程度评级：{severity_label}
- 核心临床诊断：{diagnosis_text}
- 激活特征模型：Expert_{expert_decision} (这代表了患者独特的发音声学异常特征)

【医学参考文献】
{context_text}

【你的任务】
请为该患者生成一份格式化的临床建议报告，包含以下 3 个部分（请直接输出 Markdown 内容，语气专业且温暖）：
1. **病情解读**：用通俗、温暖的语言向患者解释上述评估结果。
2. **日常护嗓指南**：结合参考文献，给出 2-3 条生活建议。
3. **康复训练动作**：结合参考文献，推荐 2 项具体的康复发声练习动作（如果文献中有的话）。
"""
            
            # 4. 调用大模型生成
            chat_completion = await llm_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "你是一位专业的言语康复理疗师，提供专业且温暖的医疗建议。"},
                    {"role": "user", "content": prompt}
                ],
                model=LLM_MODEL_NAME,
                temperature=0.7,
                max_tokens=1000
            )
            rag_report = chat_completion.choices[0].message.content
            
        except Exception as e:
            print(f"RAG 生成报告失败: {e}")
            rag_report = f"抱歉，由于知识库或生成服务异常，无法生成诊断报告。错误信息: {e}"
        finally:
            rag_time = time.time() - rag_start

    # ---------------------------------------------------------
    # 返回最终结果
    # ---------------------------------------------------------
    return JSONResponse(content={
        "status": "success",
        "file_id": file_id,
        "timing": {
            "triton_inference_sec": round(inference_time, 4),
            "rag_generation_sec": round(rag_time, 4),
            "total_sec": round(inference_time + rag_time, 4)
        },
        "prediction": {
            "class_id": pred_class,
            "severity_label": severity_label,
            "confidence": round(confidence, 4),
            "diagnosis_report": diagnosis_text
        },
        "details": {
            "router_decision": f"Expert_{expert_decision}",
            "class_probabilities": {ID2LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "router_probabilities": {f"Expert_{i}": round(float(p), 4) for i, p in enumerate(router_probs)}
        },
        "rag_clinical_advice": rag_report  # 这个就是返回给前端的最终 Markdown 报告
    })

@app.get("/health", summary="服务健康检查")
async def health_check():
    try:
        triton_client = global_state["triton_client"]
        is_live = await triton_client.is_server_live()
        return {
            "status": "healthy", 
            "triton_live": is_live,
            "rag_initialized": global_state.get("vector_store") is not None
        }
    except Exception:
        return {"status": "degraded", "triton_live": False}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("启动 FastAPI 异步后端服务 (RAG + Triton Backend)")
    print("访问 http://127.0.0.1:8088/docs 查看交互式 API 文档")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
