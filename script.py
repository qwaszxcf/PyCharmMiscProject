# 项目主脚本 - 包含请假审批规则解析和文档切割功能

import os
from leave_approval_parser import parse_leave_rules  # 假设这是现有的请假规则解析函数
from docx_chunker import process_docx_for_rag
from test_chunker import main as test_docx_chunker


def print_hi(name):
    print(f'Hi, {name}')


def run_leave_approval_parser():
    """运行请假审批规则解析功能"""
    print("运行请假审批规则解析...")
    # 这里调用现有的请假审批解析功能
    # 示例：parse_leave_rules(input_file="input.md", output_file="output.json")


def run_docx_chunker():
    """运行Word文档切割功能"""
    print("运行Word文档切割功能...")
    test_docx_chunker()


def main():
    print("欢迎使用多功能文档处理工具")
    print("1. 请假审批规则解析")
    print("2. Word文档切割（RAG预处理）")
    
    choice = input("请选择功能 (1 或 2): ")
    
    if choice == "1":
        run_leave_approval_parser()
    elif choice == "2":
        run_docx_chunker()
    else:
        print("无效选择，运行Word文档切割功能作为默认")
        run_docx_chunker()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
