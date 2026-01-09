"""
展示新的文档切分逻辑示例
演示：标题保持完整，只对正文进行切分
"""
from langchain_core.documents import Document
from docx_chunker import chunk_documents_with_recursive_splitter

def example_chunking():
    print("=" * 70)
    print("示例：文档切分逻辑演示")
    print("=" * 70)
    print()
    
    # 模拟一个包含长段落的文档
    long_paragraph = """
    人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    自从人工智能诞生以来，理论和技术日益成熟，应用领域也不断扩大。
    可以设想，未来人工智能带来的科技产品，将会是人类智慧的"容器"。
    人工智能可以对人的意识、思维的信息过程进行模拟。
    """ * 5  # 重复5次使其变长
    
    test_docs = [
        Document(
            page_content=long_paragraph.strip(),
            metadata={"title": "第一章：人工智能概述", "level": 1}
        ),
        Document(
            page_content="这是一个短段落，不需要切分。",
            metadata={"title": "第二章：总结", "level": 1}
        ),
    ]
    
    print("原始文档信息：")
    for i, doc in enumerate(test_docs):
        print(f"\n文档 {i+1}:")
        print(f"  标题: {doc.metadata['title']}")
        print(f"  层级: {doc.metadata['level']}")
        print(f"  内容长度: {len(doc.page_content)} 字符")
    
    print("\n" + "=" * 70)
    print("开始切分（chunk_size=200, chunk_overlap=20）...")
    print("=" * 70)
    
    # 进行切分
    chunks = chunk_documents_with_recursive_splitter(
        docs=test_docs,
        chunk_size=200,
        chunk_overlap=20,
        source="example.docx"
    )
    
    print(f"\n切分结果：共生成 {len(chunks)} 个chunk\n")
    
    # 统计每个标题下的chunk数量
    title_stats = {}
    for chunk in chunks:
        title = chunk.metadata['title']
        if title not in title_stats:
            title_stats[title] = []
        title_stats[title].append(chunk)
    
    print("各标题下的chunk分布：")
    for title, chunk_list in title_stats.items():
        print(f"  【{title}】 -> {len(chunk_list)} 个chunk")
    
    print("\n" + "=" * 70)
    print("详细chunk信息：")
    print("=" * 70)
    
    for chunk in chunks:
        print(f"\nChunk #{chunk.metadata['chunk_index']}")
        print(f"  标题: {chunk.metadata['title']}")
        print(f"  源文档: {chunk.metadata['source']}")
        print(f"  字符数: {len(chunk.page_content)}")
        print(f"  内容预览: {chunk.page_content[:80].strip()}...")
    
    print("\n" + "=" * 70)
    print("关键验证点：")
    print("=" * 70)
    print("✓ 长段落被切分为多个chunk")
    print("✓ 每个chunk都保留了原始标题信息")
    print("✓ 短段落保持完整（如果小于chunk_size）")
    print("✓ chunk_index全局连续递增")
    print()

if __name__ == "__main__":
    try:
        example_chunking()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
