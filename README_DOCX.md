# Word文档切割工具使用说明

## 功能介绍

本工具使用 `python-docx`、`langchain` 和 `tiktoken` 技术栈对Word文档进行智能切割，适用于RAG系统中文档的预处理。

## 依赖安装

在使用前，请先安装所需的依赖包：

```bash
pip install python-docx langchain tiktoken
```

或者使用项目中的 requirements.txt 文件：

```bash
pip install -r requirements.txt
```

## 核心功能

1. **文档解析**：使用 `python-docx` 解析 Word 文档，识别标题层级和段落内容
2. **结构化处理**：按标题层级组织文档内容，保留章节结构信息
3. **智能分块**：使用 `langchain` 的 `RecursiveCharacterTextSplitter` 进行基于字符的递归文档分块

## 使用方法

### 方法一：直接运行测试脚本

```bash
python test_chunker.py
```

该脚本会自动创建一个示例文档 `testdoc.docx` 并进行切割处理。

### 方法二：在自己的代码中使用

```python
from docx_chunker import process_docx_for_rag

# 处理文档，返回切分后的文档块列表
chunks = process_docx_for_rag("your_document.docx", chunk_size=500, chunk_overlap=50)

# 遍历处理结果
for i, chunk in enumerate(chunks):
    print(f"块 {i}: {chunk.metadata.get('title')}")
    print(f"内容: {chunk.page_content[:100]}...")
```

## 参数说明

- `chunk_size`: 每个文档块的最大字符数，默认为 500
- `chunk_overlap`: 相邻文档块之间的重叠字符数，默认为 50

## 输出格式

每个文档块都是 LangChain 的 Document 对象，包含：
- `page_content`: 文档块的实际内容
- `metadata`: 元数据信息，包含：
  - `title`: 对应的标题
  - `level`: 标题层级

## 注意事项

1. 确保 Word 文档使用了正确的标题样式（Heading 1, Heading 2 等）以便正确识别章节结构
2. 根据具体需求调整 `chunk_size` 和 `chunk_overlap` 参数
3. 使用递归字符分割策略，优先在段落、句子和单词边界处分割