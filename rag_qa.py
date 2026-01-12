"""
RAG 问答模块

功能：
1. 对接百炼平台大模型（qwen-plus）
2. 结合 FAISS 检索结果进行问答
3. 强制引用机制：每个结论必须对应 chunk_id
4. 支持 JSON Schema 结构化输出
"""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from openai import OpenAI
from faiss_indexer import FAISSIndexer


# ==================== 配置 ====================

@dataclass
class LLMConfig:
    """大模型配置"""
    model_name: str = "qwen-plus"  # 可选: qwen-turbo, qwen-max
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: float = 0.3  # 降低温度，使回答更精确
    max_tokens: int = 2048


@dataclass
class RAGConfig:
    """RAG 检索配置"""
    top_k: int = 5                    # 最大检索数量
    min_chunks: int = 3               # 最少使用 chunk 数
    max_chunks: int = 5               # 最多使用 chunk 数
    score_threshold: float = 0.35     # 相似度阈值，低于此值不进入 prompt


# ==================== Prompt 模板 ====================

RAG_SYSTEM_PROMPT = """你是一个严谨的问答助手，必须严格根据提供的参考资料回答问题。

【核心规则 - 必须遵守】
1. 每个结论必须至少引用一个资料编号（如 [chunk_0001]）
2. 禁止出现任何未引用资料的判断或结论
3. 禁止使用"综合来看"、"通常情况下"、"一般而言"等泛化表述
4. 如果资料不足以回答，必须明确说明"根据现有资料无法确定"
5. 回答必须精准、具体，直接指向资料内容

【回答格式要求】
- 使用 JSON 格式输出
- 必须包含 answer（回答内容）和 citations（引用的资料ID列表）
- citations 中只能包含实际引用过的资料ID
- 回答中的每个要点后面必须标注 [资料ID]"""

RAG_USER_PROMPT_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}

请严格根据上述参考资料回答问题。要求：
1. 每个结论后必须标注引用来源，格式：[chunk_xxx]
2. 只使用提供的资料，不要添加任何资料外的信息
3. 以 JSON 格式输出，包含 answer 和 citations 字段

输出格式示例：
{{
    "answer": "根据规定，xxx [chunk_0001]。具体要求是 xxx [chunk_0002]。",
    "citations": ["chunk_0001", "chunk_0002"]
}}"""

NO_CONTEXT_RESPONSE = {
    "answer": "抱歉，根据现有资料无法找到与您问题相关的内容。请尝试换个方式提问，或者提供更多细节。",
    "citations": []
}


# ==================== LLM 客户端 ====================

class LLMClient:
    """百炼平台大模型客户端（OpenAI 兼容接口）"""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        
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
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> str:
        """
        与大模型对话
        
        Args:
            messages: 消息列表
            stream: 是否流式输出
            
        Returns:
            模型回复文本
        """
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream
        )
        
        if stream:
            return self._handle_stream(response)
        
        return response.choices[0].message.content
    
    def _handle_stream(self, response) -> str:
        """处理流式响应"""
        full_content = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content
        print()  # 换行
        return full_content


# ==================== RAG 问答引擎 ====================

class RAGEngine:
    """RAG 问答引擎"""
    
    def __init__(
        self,
        indexer: FAISSIndexer = None,
        llm_config: LLMConfig = None,
        rag_config: RAGConfig = None
    ):
        """
        初始化 RAG 引擎
        
        Args:
            indexer: FAISS 索引器（如果为 None，将自动加载）
            llm_config: 大模型配置
            rag_config: RAG 检索配置
        """
        # 初始化或加载索引器
        if indexer is None:
            self.indexer = FAISSIndexer()
            self.indexer.load_index()
        else:
            self.indexer = indexer
        
        # 初始化配置
        self.llm_config = llm_config or LLMConfig()
        self.rag_config = rag_config or RAGConfig()
        
        # 初始化 LLM 客户端
        self.llm = LLMClient(self.llm_config)
    
    def ask(
        self, 
        question: str, 
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        RAG 问答
        
        Args:
            question: 用户问题
            stream: 是否流式输出
            
        Returns:
            {
                "answer": str,              # 回答内容（带引用标注）
                "citations": List[str],     # 引用的 chunk_id 列表
                "sources": List[Dict],      # 完整的引用来源信息
                "has_context": bool         # 是否找到相关资料
            }
        """
        # 1. 检索相关 chunk（应用阈值过滤）
        print(f"🔍 正在检索相关资料...")
        retrieved_chunks = self._retrieve_chunks(question)
        
        # 2. 处理无资料情况
        if not retrieved_chunks:
            print("⚠️ 未找到相关资料（低于相似度阈值）")
            return {
                **NO_CONTEXT_RESPONSE,
                "sources": [],
                "has_context": False
            }
        
        print(f"✅ 找到 {len(retrieved_chunks)} 个相关片段（阈值: {self.rag_config.score_threshold}）")
        for chunk in retrieved_chunks:
            print(f"   - {chunk['chunk_id']} (score: {chunk['score']})")
        
        # 3. 构建上下文
        context = self._build_context(retrieved_chunks)
        available_chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        
        # 4. 构建消息
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": RAG_USER_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )}
        ]
        
        # 5. 调用大模型
        print(f"\n🤖 正在生成回答...\n")
        raw_response = self.llm.chat(messages, stream=stream)
        
        # 6. 解析响应并验证 citations
        result = self._parse_and_validate_response(
            raw_response, 
            available_chunk_ids,
            retrieved_chunks
        )
        
        return result
    
    def _retrieve_chunks(self, question: str) -> List[Dict]:
        """
        检索相关 chunk，应用阈值和数量限制
        
        Returns:
            过滤后的 chunk 列表（3-5个，且高于阈值）
        """
        # 检索 top_k 个结果
        results = self.indexer.search_with_chunks(
            query=question,
            top_k=self.rag_config.top_k,
            score_threshold=self.rag_config.score_threshold
        )
        
        # 已经被 search_with_chunks 过滤掉低于阈值的结果
        # 限制数量在 min_chunks ~ max_chunks 之间
        if len(results) > self.rag_config.max_chunks:
            results = results[:self.rag_config.max_chunks]
        
        return results
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """将 chunk 列表构建为上下文文本"""
        context_parts = []
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            title = chunk.get("metadata", {}).get("title", "")
            text = chunk["text"]
            score = chunk["score"]
            
            # 格式：[chunk_id] (相关度: 0.xx) 标题（如有）\n内容
            header = f"[{chunk_id}] (相关度: {score})"
            if title:
                header += f" {title}"
            
            context_parts.append(f"{header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _parse_and_validate_response(
        self, 
        raw_response: str, 
        available_chunk_ids: List[str],
        retrieved_chunks: List[Dict]
    ) -> Dict[str, Any]:
        """
        解析模型响应并验证 citations
        
        Args:
            raw_response: 模型原始输出
            available_chunk_ids: 可用的 chunk_id 列表
            retrieved_chunks: 检索到的 chunk 完整信息
        """
        # 尝试解析 JSON
        try:
            # 清理可能的 markdown 代码块标记
            clean_response = raw_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            parsed = json.loads(clean_response)
            answer = parsed.get("answer", "")
            citations = parsed.get("citations", [])
            
        except json.JSONDecodeError:
            # JSON 解析失败，使用原始文本作为答案
            print("⚠️ JSON 解析失败，使用原始响应")
            answer = raw_response
            citations = []
        
        # 验证 citations：只保留存在于 available_chunk_ids 中的
        valid_citations = [
            cid for cid in citations 
            if cid in available_chunk_ids
        ]
        
        # 检查是否有无效引用
        invalid_citations = set(citations) - set(valid_citations)
        if invalid_citations:
            print(f"⚠️ 移除了无效引用: {invalid_citations}")
        
        # 构建 sources（引用来源详情）
        sources = []
        for chunk in retrieved_chunks:
            if chunk["chunk_id"] in valid_citations:
                sources.append({
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "metadata": chunk.get("metadata", {})
                })
        
        return {
            "answer": answer,
            "citations": valid_citations,
            "sources": sources,
            "has_context": True
        }


# ==================== 交互式问答 ====================

def interactive_qa():
    """交互式问答 Demo"""
    print("=" * 60)
    print("RAG 智能问答系统（带强制引用）")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'config' 查看当前配置\n")
    
    # 初始化 RAG 引擎
    rag_config = RAGConfig(
        top_k=5,
        min_chunks=3,
        max_chunks=5,
        score_threshold=0.35
    )
    
    rag = RAGEngine(rag_config=rag_config)
    
    print(f"当前配置:")
    print(f"  - 相似度阈值: {rag_config.score_threshold}")
    print(f"  - Chunk 数量: {rag_config.min_chunks}-{rag_config.max_chunks}")
    print(f"  - 模型: {rag.llm_config.model_name}")
    
    while True:
        try:
            question = input("\n📝 请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见!")
                break
            
            if question.lower() == 'config':
                print(f"\n当前配置:")
                print(f"  - 相似度阈值: {rag_config.score_threshold}")
                print(f"  - Chunk 数量: {rag_config.min_chunks}-{rag_config.max_chunks}")
                print(f"  - 模型: {rag.llm_config.model_name}")
                continue
            
            print("-" * 40)
            result = rag.ask(question, stream=False)
            
            # 显示结果
            print("\n" + "=" * 40)
            print("📋 回答:")
            print(result["answer"])
            
            print("\n📚 引用来源:")
            if result["citations"]:
                for cid in result["citations"]:
                    print(f"  - {cid}")
            else:
                print("  （无引用）")
            
            # 显示来源详情
            if result["sources"]:
                print("\n📖 来源详情:")
                for src in result["sources"]:
                    print(f"  [{src['chunk_id']}] (相关度: {src['score']})")
                    print(f"    {src['text'][:100]}...")
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    interactive_qa()
