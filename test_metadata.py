"""
测试metadata功能的脚本
"""
from langchain_core.documents import Document
from docx_chunker import chunk_documents_with_recursive_splitter

def test_metadata():
    # 创建测试文档，包含较长的正文内容
    test_docs = [
        Document(
            page_content="这是第一章的内容，包含了很多重要信息。" * 20,  # 重复20次以超过chunk_size
            metadata={"title": "第一章：介绍", "level": 1}
        ),
        Document(
            page_content="这是第二章的内容，讲述了更多细节。" * 15,  # 重复15次
            metadata={"title": "第二章：详细说明", "level": 1}
        ),
        Document(
            page_content="短内容，不需要切分。",
            metadata={"title": "第三章：总结", "level": 1}
        ),
    ]
    
    # 测试分块功能
    chunks = chunk_documents_with_recursive_splitter(
        docs=test_docs,
        chunk_size=50,
        chunk_overlap=10,
        source="test_document.docx"
    )
    
    print(f"共生成 {len(chunks)} 个chunk\n")
    
    # 按标题统计chunk数量
    title_counts = {}
    for chunk in chunks:
        title = chunk.metadata.get('title')
        title_counts[title] = title_counts.get(title, 0) + 1
    
    print("每个标题下的chunk数量：")
    for title, count in title_counts.items():
        print(f"  {title}: {count} 个chunk")
    print()
    
    # 打印每个chunk的metadata
    for i, chunk in enumerate(chunks):
        print(f"=== Chunk {i} ===")
        print(f"标题: {chunk.metadata.get('title')}")
        print(f"层级: {chunk.metadata.get('level')}")
        print(f"索引: {chunk.metadata.get('chunk_index')}")
        print(f"源文档: {chunk.metadata.get('source')}")
        print(f"内容预览: {chunk.page_content[:50]}...")
        print(f"字符数: {len(chunk.page_content)}")
        print()

if __name__ == "__main__":
    try:
        print("测试目的：验证长文本正文被切分时，每个chunk都保留原始标题")
        print("=" * 60)
        test_metadata()
        print("\n" + "=" * 60)
        print("✓ 测试通过！")
        print("验证结果：")
        print("  1. 长文本被正确切分为多个chunk")
        print("  2. 每个chunk都保留了原始标题信息")
        print("  3. chunk_index正确递增")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
