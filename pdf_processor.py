import torch
import base64
from io import BytesIO
from PIL import Image
import json

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

def process_pdf_page(model, processor, pdf_path, page_num, device, max_tokens=1024):
    """
    处理PDF文件的指定页面并返回OCR结果
    
    参数:
        model: 已加载的模型
        processor: 文本处理器
        pdf_path: PDF文件路径
        page_num: 要处理的页码(从1开始)
        device: 计算设备
        max_tokens: 生成的最大令牌数
        
    返回:
        dict: 包含OCR结果和元数据的字典
    """
    # 渲染页面为图像
    image_base64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1024)
    
    # 构建提示
    anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text)
    
    # 构建完整提示
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    
    # 应用聊天模板和处理器
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
    
    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}
    
    # 生成输出
    output = model.generate(
        **inputs,
        temperature=0.8,
        max_new_tokens=max_tokens,
        num_return_sequences=1,
        do_sample=True
    )
    
    # 解码输出
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )[0]
    
    # 尝试解析JSON
    result = {
        "page": page_num,
        "raw_output": text_output
    }
    
    try:
        parsed = json.loads(text_output)
        result["parsed"] = parsed
        result["text"] = parsed.get("natural_text", "")
    except:
        result["text"] = text_output
    
    return result

def process_full_pdf(model, processor, pdf_path, device, start_page=1, end_page=None):
    """
    处理PDF文件的多个页面
    
    参数:
        model: 已加载的模型
        processor: 文本处理器
        pdf_path: PDF文件路径
        device: 计算设备
        start_page: 起始页码
        end_page: 结束页码(如果为None，则处理所有页面)
        
    返回:
        list: 每个页面的OCR结果列表
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(pdf_path)
    if end_page is None:
        end_page = doc.page_count
    
    results = []
    for page_num in range(start_page, min(end_page + 1, doc.page_count + 1)):
        print(f"处理第 {page_num} 页...")
        result = process_pdf_page(model, processor, pdf_path, page_num, device)
        results.append(result)
        
    return results