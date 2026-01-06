import json
import os
import time
from openai import OpenAI
import jsonschema

def read_md_file(file_path):
    """读取markdown文件内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def construct_prompt(content):
    """构造提示词"""
    prompt = f"""
    你是一个企业规则解析助手。
    请根据下面的文档内容，提取审批规则。
    
    要求：
    - 严格返回 JSON
    - 不要输出任何解释性文字
    - 无法确定的字段返回 null
    
    
    JSON 格式如下：
    {{
      "rules": [
        {{
          "condition": "string",
          "approver": "string",
          "remark": "string | null"
        }}
      ]
    }}
    文档内容
    {content}
    
    
    请只返回JSON格式的数据，不要包含任何其他说明文字。
    """
    return prompt


# 定义JSON Schema
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "approver": {"type": "string"},
                    "remark": {"type": ["string", "null"]}
                },
                "required": ["condition", "approver", "remark"],
                "additionalProperties": False
            }
        }
    },
    "required": ["rules"],
    "additionalProperties": False
}


def validate_json_schema(data):
    """校验JSON数据是否符合预定义的Schema"""
    try:
        jsonschema.validate(data, JSON_SCHEMA)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, str(e)


def construct_corrective_prompt(original_content, error_type, error_details, original_result=None):
    """构建修正提示，在重试时指出错误并指导模型生成正确格式"""
    base_prompt = f"""
    你是一个企业规则解析助手。
    请根据下面的文档内容，提取审批规则。
    
    要求：
    - 严格返回 JSON
    - 不要输出任何解释性文字
    - 无法确定的字段返回 null
    
    
    JSON 格式如下：
    {{
      "rules": [
        {{
          "condition": "string",
          "approver": "string",
          "remark": "string | null"
        }}
      ]
    }}
    
    文档内容：
    {original_content}
    
    
    重要：
    - 你之前的响应出现了 {error_type}：{error_details}
    """
    
    if original_result is not None:
        base_prompt += f"""
    - 你之前的输出是：
    {original_result}
    
    请仔细检查输出格式，确保完全符合上述JSON格式要求，不要添加任何解释性文字。
    """
    else:
        base_prompt += f"""
    请仔细检查输出格式，确保完全符合上述JSON格式要求，不要添加任何解释性文字。
    """
        
    base_prompt += """
    请只返回JSON格式的数据，不要包含任何其他说明文字。
    """
    
    return base_prompt


def call_openai_api_with_validation(content, temperature=0):
    """调用OpenAI API并进行JSON Schema校验，带重试机制"""
    max_retries = 2
    current_content = content  # 保存当前的content，用于重试时可能的更新
    
    for attempt in range(max_retries + 1):  # 总共尝试1次 + 重试2次 = 3次
        print(f"正在调用API... (第 {attempt + 1} 次尝试)")
        
        # 使用固定的temperature=0以确保输出的一致性
        current_temperature = temperature
            
        try:
            result = call_openai_api(current_content, temperature=current_temperature)
            
            # 尝试解析返回的JSON
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print("返回内容无法解析为JSON，将进行重试")
                if attempt < max_retries:
                    # 在重试时更新prompt，指出错误
                    current_content = construct_corrective_prompt(content, "JSON解析错误", str(e))
                    print(f"等待2秒后重试...")
                    time.sleep(2)
                    continue
                else:
                    # 最终失败，进行兜底处理
                    print("API返回结果无法解析为JSON格式，进行兜底处理")
                    fallback_handling(content, result)
                    return None
            
            # 校验JSON Schema
            is_valid, error_msg = validate_json_schema(parsed_result)
            
            if is_valid:
                print("JSON校验通过")
                return parsed_result
            else:
                print(f"JSON校验失败: {error_msg}")
                if attempt < max_retries:
                    # 在重试时更新prompt，指出具体的schema错误
                    current_content = construct_corrective_prompt(content, "JSON Schema校验失败", error_msg, result)
                    print(f"等待2秒后重试...")
                    time.sleep(2)
                    continue
                else:
                    # 最终失败，进行兜底处理
                    print("JSON校验最终失败，进行兜底处理")
                    fallback_handling(content, result)
                    return None
                    
        except Exception as e:
            print(f"API调用错误 (第 {attempt + 1} 次尝试): {e}")
            if attempt < max_retries:
                # 在重试时更新prompt，指出API调用错误
                current_content = construct_corrective_prompt(content, "API调用错误", str(e))
                print(f"等待2秒后重试...")
                time.sleep(2)
                continue
            else:
                print("API调用最终失败，进行兜底处理")
                fallback_handling(content, result if 'result' in locals() else "API调用失败", is_api_error=True)
                return None
    return None


def fallback_handling(original_content, raw_response, is_api_error=False):
    """兜底处理：记录原始文本，便于人工介入"""
    fallback_data = {
        "timestamp": time.time(),
        "original_content": original_content,
        "raw_response": raw_response,
        "is_api_error": is_api_error
    }
    
    # 保存到fallback.json文件
    with open("fallback.json", "w", encoding="utf-8") as f:
        json.dump(fallback_data, f, ensure_ascii=False, indent=2)
    
    print("已记录原始内容和API返回到 fallback.json，需要人工介入处理")

def call_openai_api(prompt, temperature=0):
    """调用OpenAI API"""
    # 从环境变量获取API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未找到DASHSCOPE_API_KEY环境变量")
        print("请按以下步骤设置API密钥:")
        print("1. 获取您的OpenAI API密钥")
        print("2. 设置环境变量: export DASHSCOPE_API_KEY='your-api-key' (Linux/Mac) 或 set ODASHSCOPE_API_KEY=your-api-key (Windows)")
        print("或者在代码中直接设置(不推荐): 将api_key直接赋值为您的密钥")
        raise ValueError("请设置DASHSCOPE_API_KEY环境变量")
    
    client = OpenAI(api_key=api_key,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "你是一个严谨的 JSON 生成器"},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,  # 添加temperature参数
        response_format={ "type": "json_object" }  # 请求JSON格式响应
    )
    
    return response.choices[0].message.content

def parse_and_save_result(result, output_path):
    """解析结果并保存到文件"""
    try:
        # 解析API返回的JSON字符串
        parsed_result = json.loads(result)
        
        # 保存到output.json
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(parsed_result, file, ensure_ascii=False, indent=2)
        
        print("解析结果已保存到output.json")
        print("解析的内容如下：")
        print(json.dumps(parsed_result, ensure_ascii=False, indent=2))
        
        return parsed_result
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("API返回的内容：")
        print(result)
        return None

def main():
    input_file = "input.md"
    output_file = "output.json"
    
    # 1. 读取input.md
    print("正在读取input.md...")
    content = read_md_file(input_file)
    print("读取完成")
    
    # 2. 构造提示词
    print("正在构造提示词...")
    prompt = construct_prompt(content)
    
    # 3. 调用DashScope API并进行JSON校验和重试
    print("正在调用DashScope API并进行JSON校验...")
    try:
        result = call_openai_api_with_validation(prompt)
        
        if result is not None:
            print("API调用和校验成功")
            
            # 4. 打印返回结果
            print("\nAPI返回结果：")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 5. 保存结果
            print("\n正在保存结果...")
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            print("解析结果已保存到output.json")
            print("解析的内容如下：")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("API调用和校验失败，已记录到fallback.json，需要人工介入处理")
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()