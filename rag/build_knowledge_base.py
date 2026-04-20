import os
import argparse
from typing import List

# LangChain 相关依赖
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore

# Qdrant 客户端
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def load_documents(data_dir: str):
    """
    遍历指定目录，加载支持的文档 (PDF, TXT, DOCX)
    """
    documents = []
    supported_extensions = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader
    }

    print(f"📂 正在扫描目录: {data_dir}...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"⚠️ 目录 {data_dir} 不存在，已自动创建。请将文献放入该目录后重试。")
        return documents

    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                print(f"📄 正在加载: {file}")
                loader_cls = supported_extensions[ext]
                try:
                    loader = loader_cls(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"❌ 加载 {file} 失败: {e}")
            else:
                print(f"⏭️ 跳过不支持的文件: {file}")
                
    print(f"✅ 共加载 {len(documents)} 页/篇 文档内容。")
    return documents


def build_knowledge_base(data_dir: str, collection_name: str, qdrant_url: str):
    """
    读取文档，切分并向量化，最后存入 Qdrant。
    """
    docs = load_documents(data_dir)
    if not docs:
        print("没有找到任何文档，流程终止。")
        return

    # 医疗文献通常句子较长，需要保留上下文。这里按 500 字一块进行切分，重叠 100 字。
    print("✂️ 正在进行文本切分...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ 文档已被切分为 {len(chunks)} 个文本块 (Chunks)。")

    # BAAI/bge-large-zh-v1.5 是目前开源中文 Embedding 模型的 SOTA（State of the Art），非常适合医疗等垂直领域。
    print("🧠 正在加载 BGE 向量模型 (首次运行会自动下载模型权重，请耐心等待)...")
    model_name = "BAAI/bge-large-zh-v1.5"
    model_kwargs = {'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # BGE 模型推荐设置为 True
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print(f"🔌 正在连接 Qdrant 数据库: {qdrant_url} ...")
    client = QdrantClient(url=qdrant_url, timeout=60.0)
    
    # 获取 BGE 模型的维度，bge-large-zh-v1.5 的维度是 1024
    vector_size = 1024 
    
    # 检查集合是否存在，不存在则创建
    if not client.collection_exists(collection_name):
        print(f"🏗️ 正在创建 Qdrant Collection: '{collection_name}' (维度: {vector_size})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        print(f"ℹ️ Collection '{collection_name}' 已存在，即将向其中追加数据。")

    print("🚀 正在将文本块向量化并写入 Qdrant 数据库...")
    # 使用 from_documents 快捷方法，它会自动调用 embeddings 模型将 chunks 转化为向量并写入
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=qdrant_url,
        collection_name=collection_name,
        # 强制复用刚刚确认/创建的 collection
        force_recreate=False 
    )
    print("🎉 知识库构建完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建言语康复知识库并导入 Qdrant")
    parser.add_argument("--data_dir", type=str, default="rag/data", help="存放医学文献(PDF/TXT/DOCX)的目录")
    parser.add_argument("--collection", type=str, default="speech_rehab_kb", help="Qdrant 中的集合名称")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333", help="Qdrant 服务的 REST API 地址")
    
    args = parser.parse_args()
    
    build_knowledge_base(
        data_dir=args.data_dir,
        collection_name=args.collection,
        qdrant_url=args.qdrant_url
    )
