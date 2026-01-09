"""
FAISS 向量索引与检索模块

功能：
1. 使用云端 Embedding（百炼平台，OpenAI 兼容接口）生成向量
2. 使用 FAISS IndexFlatIP 存储向量（配合 L2 归一化实现余弦相似度）
3. FAISS 内部 ID 与 chunk_id 分离管理
4. 索引文件管理：faiss.index, chunks.jsonl, id_map.json, embedding_meta.json
"""

import os
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import faiss
from openai import OpenAI


# ==================== 配置 ====================

@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    model_name: str = "text-embedding-v3"
    dimension: int = 1024
    normalize: bool = True  # 是否 L2 归一化（余弦相似度必须为 True）
    batch_size: int = 10
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class IndexConfig:
    """索引配置"""
    index_type: str = "IndexFlatIP"  # IndexFlatIP 或 IndexFlatL2
    index_dir: str = "faiss_store"   # 索引文件存储目录
    
    # 索引文件名
    index_file: str = "faiss.index"
    chunks_file: str = "chunks.jsonl"
    id_map_file: str = "id_map.json"
    embedding_meta_file: str = "embedding_meta.json"


# ==================== Embedding 客户端 ====================

class EmbeddingClient:
    """云端 Embedding 客户端（百炼平台，OpenAI 兼容接口）"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        
        # 从环境变量获取 API 密钥
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "未找到 DASHSCOPE_API_KEY 环境变量\n"
                "请设置: $env:DASHSCOPE_API_KEY='your-api-key' (PowerShell)"
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.api_base_url
        )
    
    def embed_texts(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        """
        批量生成文本向量
        
        Args:
            texts: 文本列表
            max_retries: 最大重试次数
            
        Returns:
            np.ndarray: 向量矩阵，形状 (n, dimension)，dtype=float32
        """
        all_embeddings = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch_texts, max_retries)
            all_embeddings.extend(batch_embeddings)
            
            # 打印进度
            processed = min(i + batch_size, len(texts))
            print(f"Embedding 进度: {processed}/{len(texts)}")
        
        # 转换为 numpy 数组
        embeddings = np.array(all_embeddings, dtype='float32')
        
        # L2 归一化（如果配置要求）
        if self.config.normalize:
            faiss.normalize_L2(embeddings)
            print("已完成 L2 归一化")
        
        return embeddings
    
    def _embed_batch_with_retry(self, texts: List[str], max_retries: int) -> List[List[float]]:
        """带重试的批量 Embedding"""
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model_name,
                    input=texts
                )
                # 按 index 排序确保顺序正确
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
                
            except Exception as e:
                print(f"Embedding 请求失败 (第 {attempt + 1} 次): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Embedding 请求最终失败: {e}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        生成查询向量（单条）
        
        Args:
            query: 查询文本
            
        Returns:
            np.ndarray: 向量，形状 (1, dimension)，dtype=float32
        """
        embeddings = self.embed_texts([query])
        return embeddings


# ==================== FAISS 索引管理 ====================

class FAISSIndexer:
    """FAISS 向量索引管理器"""
    
    def __init__(
        self,
        embedding_config: EmbeddingConfig = None,
        index_config: IndexConfig = None
    ):
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.index_config = index_config or IndexConfig()
        
        self.embedding_client = EmbeddingClient(self.embedding_config)
        
        # 运行时数据
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []           # 原始 chunk 数据
        self.id_map: Dict[int, str] = {}       # faiss_id -> chunk_id
        self.reverse_id_map: Dict[str, int] = {}  # chunk_id -> faiss_id
        self.next_faiss_id: int = 0
    
    # -------------------- 索引构建 --------------------
    
    def build_index(self, chunks: List[Dict]) -> None:
        """
        从 chunk 列表构建索引
        
        Args:
            chunks: chunk 列表，每个 chunk 必须包含:
                - chunk_id: str, 唯一标识
                - text: str, 文本内容
                - metadata: dict, 元数据（可选）
        """
        if not chunks:
            raise ValueError("chunks 列表不能为空")
        
        print(f"开始构建索引，共 {len(chunks)} 个 chunk")
        
        # 1. 提取文本
        texts = [chunk["text"] for chunk in chunks]
        
        # 2. 生成向量（已包含归一化）
        print("正在生成 Embedding...")
        embeddings = self.embedding_client.embed_texts(texts)
        
        # 3. 创建 FAISS 索引
        dimension = embeddings.shape[1]
        if self.index_config.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
            print(f"创建 IndexFlatIP 索引，维度: {dimension}")
        else:
            self.index = faiss.IndexFlatL2(dimension)
            print(f"创建 IndexFlatL2 索引，维度: {dimension}")
        
        # 4. 添加向量到索引
        self.index.add(embeddings)
        
        # 5. 构建 ID 映射
        self.chunks = chunks
        self.id_map = {}
        self.reverse_id_map = {}
        
        for faiss_id, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            self.id_map[faiss_id] = chunk_id
            self.reverse_id_map[chunk_id] = faiss_id
        
        self.next_faiss_id = len(chunks)
        
        print(f"索引构建完成，共 {self.index.ntotal} 个向量")
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        增量添加 chunk 到已有索引
        
        Args:
            chunks: 新的 chunk 列表
        """
        if self.index is None:
            raise RuntimeError("索引未初始化，请先调用 build_index 或 load_index")
        
        if not chunks:
            return
        
        print(f"增量添加 {len(chunks)} 个 chunk")
        
        # 生成向量
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_client.embed_texts(texts)
        
        # 添加到索引
        self.index.add(embeddings)
        
        # 更新 ID 映射
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            faiss_id = self.next_faiss_id
            
            self.id_map[faiss_id] = chunk_id
            self.reverse_id_map[chunk_id] = faiss_id
            self.chunks.append(chunk)
            
            self.next_faiss_id += 1
        
        print(f"增量添加完成，当前共 {self.index.ntotal} 个向量")
    
    # -------------------- 检索 --------------------
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        检索与查询最相似的 chunk（TopK + 阈值双保险）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量（默认 5）
            score_threshold: 相似度阈值（可选），低于此阈值的结果将被过滤
            
        Returns:
            检索结果列表，格式:
            [
                {"chunk_id": "doc1_chunk_03", "score": 0.82},
                {"chunk_id": "doc2_chunk_01", "score": 0.79},
                ...
            ]
            如果所有结果都低于阈值，返回空列表
        """
        if self.index is None:
            raise RuntimeError("索引未初始化，请先调用 build_index 或 load_index")
        
        # 生成查询向量
        query_embedding = self.embedding_client.embed_query(query)
        
        # FAISS 检索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for i, (faiss_id, score) in enumerate(zip(indices[0], distances[0])):
            if faiss_id == -1:  # FAISS 返回 -1 表示无效结果
                continue
            
            # 阈值过滤（如果设置了阈值）
            if score_threshold is not None and score < score_threshold:
                continue
            
            chunk_id = self.id_map.get(int(faiss_id))
            if chunk_id:
                results.append({
                    "chunk_id": chunk_id,
                    "score": round(float(score), 4)
                })
        
        return results
    
    def search_with_chunks(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        检索并返回完整的 chunk 信息（TopK + 阈值双保险）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量（默认 5）
            score_threshold: 相似度阈值（可选），低于此阈值的结果将被过滤
            
        Returns:
            检索结果列表，包含完整 chunk 信息
            如果所有结果都低于阈值，返回空列表
        """
        if self.index is None:
            raise RuntimeError("索引未初始化")
        
        # 生成查询向量
        query_embedding = self.embedding_client.embed_query(query)
        
        # FAISS 检索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 构建结果（包含完整 chunk）
        results = []
        for faiss_id, score in zip(indices[0], distances[0]):
            if faiss_id == -1:
                continue
            
            # 阈值过滤（如果设置了阈值）
            if score_threshold is not None and score < score_threshold:
                continue
            
            faiss_id = int(faiss_id)
            chunk_id = self.id_map.get(faiss_id)
            
            if chunk_id and faiss_id < len(self.chunks):
                chunk = self.chunks[faiss_id]
                results.append({
                    "chunk_id": chunk_id,
                    "score": round(float(score), 4),
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {})
                })
        
        return results
    
    # -------------------- 持久化 --------------------
    
    def save_index(self, index_dir: str = None) -> None:
        """
        保存索引到磁盘
        
        生成文件:
        - faiss.index: FAISS 向量索引
        - chunks.jsonl: 原始 chunk 数据
        - id_map.json: faiss_id <-> chunk_id 映射
        - embedding_meta.json: 模型信息
        """
        if self.index is None:
            raise RuntimeError("索引未初始化")
        
        index_dir = index_dir or self.index_config.index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # 1. 保存 FAISS 索引
        index_path = os.path.join(index_dir, self.index_config.index_file)
        faiss.write_index(self.index, index_path)
        print(f"已保存 FAISS 索引: {index_path}")
        
        # 2. 保存 chunks (JSONL 格式)
        chunks_path = os.path.join(index_dir, self.index_config.chunks_file)
        with open(chunks_path, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        print(f"已保存 chunks: {chunks_path}")
        
        # 3. 保存 ID 映射
        id_map_path = os.path.join(index_dir, self.index_config.id_map_file)
        id_map_data = {
            "faiss_to_chunk": {str(k): v for k, v in self.id_map.items()},
            "chunk_to_faiss": self.reverse_id_map,
            "next_faiss_id": self.next_faiss_id
        }
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_map_data, f, ensure_ascii=False, indent=2)
        print(f"已保存 ID 映射: {id_map_path}")
        
        # 4. 保存 Embedding 元信息
        meta_path = os.path.join(index_dir, self.index_config.embedding_meta_file)
        meta_data = {
            "model_name": self.embedding_config.model_name,
            "dimension": self.embedding_config.dimension,
            "normalize": self.embedding_config.normalize,
            "index_type": self.index_config.index_type,
            "total_vectors": self.index.ntotal,
            "created_at": datetime.now().isoformat(),
            "api_base_url": self.embedding_config.api_base_url
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        print(f"已保存 Embedding 元信息: {meta_path}")
        
        print(f"索引保存完成，目录: {index_dir}")
    
    def load_index(self, index_dir: str = None) -> None:
        """
        从磁盘加载索引
        """
        index_dir = index_dir or self.index_config.index_dir
        
        # 1. 加载 FAISS 索引
        index_path = os.path.join(index_dir, self.index_config.index_file)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"已加载 FAISS 索引: {index_path}，共 {self.index.ntotal} 个向量")
        
        # 2. 加载 chunks
        chunks_path = os.path.join(index_dir, self.index_config.chunks_file)
        self.chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))
        print(f"已加载 chunks: {len(self.chunks)} 条")
        
        # 3. 加载 ID 映射
        id_map_path = os.path.join(index_dir, self.index_config.id_map_file)
        with open(id_map_path, 'r', encoding='utf-8') as f:
            id_map_data = json.load(f)
        
        self.id_map = {int(k): v for k, v in id_map_data["faiss_to_chunk"].items()}
        self.reverse_id_map = id_map_data["chunk_to_faiss"]
        self.next_faiss_id = id_map_data["next_faiss_id"]
        print(f"已加载 ID 映射: {len(self.id_map)} 条")
        
        # 4. 加载 Embedding 元信息（用于验证）
        meta_path = os.path.join(index_dir, self.index_config.embedding_meta_file)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        # 验证模型一致性
        if meta_data["model_name"] != self.embedding_config.model_name:
            print(f"警告: 当前模型 ({self.embedding_config.model_name}) "
                  f"与索引模型 ({meta_data['model_name']}) 不一致")
        
        print(f"索引加载完成，模型: {meta_data['model_name']}")


# ==================== 工具函数 ====================

def convert_langchain_docs_to_chunks(
    docs: List[Any],
    doc_id_prefix: str = "doc"
) -> List[Dict]:
    """
    将 LangChain Document 列表转换为 chunk 列表
    
    Args:
        docs: LangChain Document 列表
        doc_id_prefix: chunk_id 前缀
        
    Returns:
        chunk 列表，格式: [{"chunk_id": str, "text": str, "metadata": dict}, ...]
    """
    chunks = []
    for i, doc in enumerate(docs):
        chunk_id = f"{doc_id_prefix}_chunk_{i:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": doc.page_content,
            "metadata": dict(doc.metadata) if hasattr(doc, 'metadata') else {}
        })
    return chunks


# ==================== 示例用法 ====================

if __name__ == "__main__":
    # 从 docx_chunker 输出的 JSON 文件构建 FAISS 索引
    
    import sys
    from pathlib import Path
    
    # 默认使用 testdoc_chunks.json
    chunks_json_file = "testdoc_chunks.json"
    
    # 检查文件是否存在
    if not Path(chunks_json_file).exists():
        print(f"错误: 找不到 {chunks_json_file}")
        print("请先运行 docx_chunker.py 生成 chunk JSON 文件")
        print("命令: python docx_chunker.py")
        sys.exit(1)
    
    try:
        print("=" * 60)
        print("从 docx_chunker 输出加载 chunks 并构建 FAISS 索引")
        print("=" * 60)
        
        # 1. 加载 chunks
        print(f"\n正在加载 {chunks_json_file}...")
        with open(chunks_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        metadata = data.get("metadata", {})
        
        print(f"加载完成:")
        print(f"  - 源文档: {metadata.get('source_file', 'N/A')}")
        print(f"  - 总 chunks: {metadata.get('total_chunks', len(chunks))}")
        print(f"  - 创建时间: {metadata.get('created_at', 'N/A')}")
        
        if not chunks:
            print("错误: chunks 列表为空")
            sys.exit(1)
        
        # 2. 创建索引器并构建索引
        print("\n" + "=" * 60)
        print("构建 FAISS 索引")
        print("=" * 60)
        
        indexer = FAISSIndexer()
        indexer.build_index(chunks)
        
        # 3. 保存索引
        print("\n" + "=" * 60)
        print("保存索引")
        print("=" * 60)
        indexer.save_index()
        
        # 4. 测试检索（TopK + 阈值双保险）
        print("\n" + "=" * 60)
        print("测试检索（TopK + 阈值双保险）")
        print("=" * 60)
        
        test_queries = [
            ("基于网络的架构和基于分布式的架构的区别", 5, 0.3),  # query, top_k, threshold
            ("完全不相关的内容测试", 5, 0.5),
        ]
        
        for query, top_k, threshold in test_queries:
            print(f"\n查询: {query}")
            print(f"  TopK: {top_k}, 阈值: {threshold}")
            
            results = indexer.search(query, top_k=top_k, score_threshold=threshold)
            
            if results:
                print(f"  找到 {len(results)} 个结果:")
                print(json.dumps(results, ensure_ascii=False, indent=4))
            else:
                print("  ⚠️ 无资料（所有结果均低于阈值）")
        
        # 5. 测试重新加载索引
        print("\n" + "=" * 60)
        print("测试重新加载索引")
        print("=" * 60)
        
        new_indexer = FAISSIndexer()
        new_indexer.load_index()
        
        query = "基于网络的架构和基于分布式的架构的区别"
        print(f"\n查询: {query}")
        results = new_indexer.search(query, top_k=5, score_threshold=0.3)
        
        if results:
            print(f"找到 {len(results)} 个结果:")
            print(json.dumps(results, ensure_ascii=False, indent=4))
        else:
            print("⚠️ 无资料")
        
        print("\n" + "=" * 60)
        print("✅ 测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
