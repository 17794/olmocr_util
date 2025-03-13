import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from pdf_processor import process_pdf_page, process_full_pdf
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(description='PDF OCR处理工具')
    parser.add_argument('--pdf_path',default="paper.pdf", help='PDF文件路径',)
    parser.add_argument('--page', type=int, default=1, help='处理特定页码(默认为1)')
    parser.add_argument('--all', action='store_true', help='处理所有页面',default=True)
    parser.add_argument('--start', type=int, default=1, help='起始页码(与--all一起使用)')
    parser.add_argument('--end', type=int, default=None, help='结束页码(与--all一起使用)')
    parser.add_argument('--output', default='ocr_results.json', help='输出文件名')
    args = parser.parse_args()
    
    # 初始化模型
    print("加载模型中...")
    model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf_path):
        print(f"错误: 找不到PDF文件 '{args.pdf_path}'")
        return
    
    # 处理PDF
    if args.all:
        print(f"处理PDF的所有页面，从第{args.start}页开始...")
        results = process_full_pdf(model, processor, args.pdf_path, device, 
                                   start_page=args.start, end_page=args.end)
        print(f"已完成。处理了{len(results)}个页面。")
    else:
        print(f"处理PDF的第{args.page}页...")
        result = process_pdf_page(model, processor, args.pdf_path, args.page, device)
        results = [result]
        print("处理完成。")
    
    # 打印结果摘要
    for i, result in enumerate(results):
        page_num = result['page']
        text = result.get('text', '')
        print(f"\n第 {page_num} 页 - 文本长度: {len(text)}")
        print(f"前100个字符: {text[:100]}...")
    
    # 保存结果到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 {args.output}")

if __name__ == "__main__":
    main()
