# 请假审批规则解析器

这是一个使用AI API解析请假审批规则的Python项目，能够从Markdown文档中提取结构化的审批规则。

## 项目概述

本项目主要功能是从Markdown格式的请假审批规则文档中提取结构化信息，使用大语言模型API进行解析并输出JSON格式的规则数据。项目利用阿里云DashScope的qwen-plus模型进行文本解析，通过OpenAI SDK的兼容模式实现API调用。

## 文件结构

- `leave_approval_parser.py` - 核心解析脚本，负责读取Markdown文件、调用API解析规则并保存结果
- `input.md` - 输入文件，包含原始的请假审批规则
- `output.json` - 输出文件，包含解析后的结构化规则
- `requirements.txt` - 项目依赖
- `script.py` - 示例Python脚本
- `helloWord.py` - Hello World示例
- `notebook.ipynb` - Jupyter Notebook示例文件

## 功能特性

1. **文档解析** - 读取Markdown格式的审批规则文档
2. **AI解析** - 使用OpenAI API（通过阿里云DashScope兼容模式）解析文档内容
3. **结构化输出** - 将解析结果保存为JSON格式
4. **规则提取** - 提取条件、审批人和备注等信息
5. **错误处理** - 包含重试机制和JSON Schema校验
6. **兜底机制** - 当解析失败时，记录原始内容便于人工处理

## 技术栈

- Python 3.x
- OpenAI Python SDK
- jsonschema库（用于数据校验）
- 阿里云DashScope API（兼容OpenAI格式）
- qwen-plus模型

## 依赖安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 设置环境变量：
   ```bash
   export DASHSCOPE_API_KEY='your-api-key'  # Linux/Mac
   # 或
   set DASHSCOPE_API_KEY=your-api-key       # Windows
   # 在PowerShell中:
   $env:DASHSCOPE_API_KEY="your-api-key"
   ```

2. 准备输入文件 `input.md`，格式如：
   ```markdown
   # 请假审批规则
   1. 请假 3 天以内，直属领导审批
   2. 超过 3 天，需要部门负责人审批
   3. 超过 7 天，需要 HR 复核
   4. 超过 14 天，需要发送邮件通知公司经理 报备
   ```

3. 运行解析脚本：
   ```bash
   python leave_approval_parser.py
   ```

4. 查看输出结果 `output.json`

## 示例输出

```json
{
  "rules": [
    {
      "condition": "请假 3 天以内",
      "approver": "直属领导",
      "remark": null
    },
    {
      "condition": "超过 3 天",
      "approver": "部门负责人",
      "remark": null
    },
    {
      "condition": "超过 7 天",
      "approver": "HR",
      "remark": "复核"
    },
    {
      "condition": "超过 14 天",
      "approver": "公司经理",
      "remark": "发送邮件通知报备"
    }
  ]
}
```

## 配置说明

- 项目使用阿里云DashScope的qwen-plus模型进行文本解析
- API调用通过OpenAI SDK的兼容模式实现
- 解析结果严格按照指定的JSON格式返回
- 包含JSON Schema校验确保输出格式正确
- 实现了重试机制应对API调用失败或格式错误

## 注意事项

- 需要有效的API密钥才能调用解析服务
- 输入文档格式应清晰，以便AI模型准确解析
- 项目当前专注于请假审批规则的解析，可根据需要扩展到其他类型的规则文档
- 当API解析失败时，系统会将原始内容和错误信息保存到fallback.json文件中
- 推荐使用temperature=0以确保输出的一致性