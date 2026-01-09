"""
RAG 完整流程示例

展示如何：
1. 从 DOCX 切分文档并冻结为 chunk JSON
2. 构建 FAISS 向量索引
3. 使用 TopK + 阈值双保险进行检索
"""

import json
from pathlib import Path

from docx_chunker import process_and_save_chunks, load_chunks_from_json
from faiss_indexer import FAISSIndexer


def main():
    # ==================== 步骤 1: 切分文档 ====================
    print("=" * 70)
    print("步骤 1: 从 DOCX 切分文档并冻结为 chunk JSON")
    print("=" * 70)
    
    docx_file = "testdoc.docx"
    chunks_json = "testdoc_chunks.json"
    
    # 检查 docx 文件是否存在
    if not Path(docx_file).exists():
        print(f"错误: 找不到 {docx_file}")
        print("请在当前目录放置 testdoc.docx 文件")
        return
    
    # 如果 chunks JSON 已存在，询问是否重新生成
    if Path(chunks_json).exists():
        print(f"发现已存在的 {chunks_json}")
        response = input("是否重新生成？(y/n): ").strip().lower()
        if response != 'y':
            print(f"使用现有的 {chunks_json}")
        else:
            chunks = process_and_save_chunks(
                file_path=docx_file,
                output_path=chunks_json,
                chunk_size=500,
                chunk_overlap=50
            )
            print(f"✅ 已重新生成 {len(chunks)} 个 chunks")
    else:
        chunks = process_and_save_chunks(
            file_path=docx_file,
            output_path=chunks_json,
            chunk_size=500,
            chunk_overlap=50
        )
        print(f"✅ 已生成 {len(chunks)} 个 chunks")
    
    # ==================== 步骤 2: 构建 FAISS 索引 ====================
    print("\n" + "=" * 70)
    print("步骤 2: 构建 FAISS 向量索引")
    print("=" * 70)
    
    # 加载 chunks
    chunks = load_chunks_from_json(chunks_json)
    
    # 创建索引器并构建索引
    indexer = FAISSIndexer()
    indexer.build_index(chunks)
    
    # 保存索引
    indexer.save_index()
    print("✅ FAISS 索引构建并保存完成")
    
    # ==================== 步骤 3: 测试检索 ====================
    print("\n" + "=" * 70)
    print("步骤 3: 测试检索（TopK + 阈值双保险）")
    print("=" * 70)
    
    # 测试查询列表
    test_queries = [
        {
            "query": "请假5天谁来审批",
            "top_k": 5,
            "threshold": 0.3,
            "description": "正常查询，预期有结果"
        },
        {
            "query": "完全不相关的查询内容测试xyz",
            "top_k": 5,
            "threshold": 0.5,
            "description": "不相关查询，预期返回'无资料'"
        },
        {
            "query": "请假超过7天需要谁审批",
            "top_k": 3,
            "threshold": 0.4,
            "description": "具体场景查询"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n【测试 {i}】{test['description']}")
        print(f"查询: {test['query']}")
        print(f"TopK: {test['top_k']}, 阈值: {test['threshold']}")
        
        results = indexer.search(
            query=test['query'],
            top_k=test['top_k'],
            score_threshold=test['threshold']
        )
        
        if results:
            print(f"✅ 找到 {len(results)} 个相关结果:")
            for j, result in enumerate(results, 1):
                print(f"  {j}. {result['chunk_id']} (score: {result['score']})")
        else:
            print("⚠️ 无资料（所有结果均低于阈值）")
    
    # ==================== 步骤 4: 展示完整 chunk 内容 ====================
    print("\n" + "=" * 70)
    print("步骤 4: 获取完整 chunk 内容（用于 RAG）")
    print("=" * 70)
    
    query = "请假5天谁来审批"
    print(f"\n查询: {query}")
    
    results_with_chunks = indexer.search_with_chunks(
        query=query,
        top_k=3,
        score_threshold=0.3
    )
    
    if results_with_chunks:
        print(f"\n找到 {len(results_with_chunks)} 个相关 chunk:\n")
        for i, result in enumerate(results_with_chunks, 1):
            print(f"【Chunk {i}】")
            print(f"  ID: {result['chunk_id']}")
            print(f"  Score: {result['score']}")
            print(f"  标题: {result['metadata'].get('title', 'N/A')}")
            print(f"  内容: {result['text'][:100]}...")
            print()
    else:
        print("⚠️ 无资料")
    
    print("=" * 70)
    print("✅ RAG 流程演示完成")
    print("=" * 70)
    print("\n说明:")
    print("1. chunk JSON 已保存到:", chunks_json)
    print("2. FAISS 索引已保存到: faiss_store/ 目录")
    print("3. 包含文件: faiss.index, chunks.jsonl, id_map.json, embedding_meta.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
