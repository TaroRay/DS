# ✅ 修改後 getPDF.py：套用與 multiagent.py 相同的 Gemini 初始化方式

import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
from google.generativeai import configure, GenerativeModel
from flask import abort

# 載入環境變數並設定 Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
configure(api_key=api_key)
model = GenerativeModel("gemini-1.5-flash")

def get_chinese_font_file() -> str:
    fonts_path = r"C:\\Windows\\Fonts"
    candidates = ["kaiu.ttf"]
    for font in candidates:
        font_path = os.path.join(fonts_path, font)
        if os.path.exists(font_path):
            print("找到系統中文字型：", font_path)
            return os.path.abspath(font_path)
    print("未在系統中找到候選中文字型檔案。")
    return None

def create_table(pdf: FPDF, df: pd.DataFrame):
    available_width = pdf.w - 2 * pdf.l_margin
    num_columns = len(df.columns)
    col_width = available_width / num_columns
    cell_height = 10

    pdf.set_fill_color(200, 200, 200)
    pdf.set_font("ChineseFont", "", 12)
    for col in df.columns:
        pdf.cell(col_width, cell_height, str(col), border=1, align="C", fill=True)
    pdf.ln(cell_height)

    pdf.set_font("ChineseFont", "", 12)
    fill = False
    for index, row in df.iterrows():
        if pdf.get_y() + cell_height > pdf.h - pdf.b_margin:
            pdf.add_page()
            pdf.set_fill_color(200, 200, 200)
            pdf.set_font("ChineseFont", "", 12)
            for col in df.columns:
                pdf.cell(col_width, cell_height, str(col), border=1, align="C", fill=True)
            pdf.ln(cell_height)
        pdf.set_fill_color(230, 240, 255 if fill else 255)
        for item in row:
            pdf.cell(col_width, cell_height, str(item), border=1, align="C", fill=True)
        pdf.ln(cell_height)
        fill = not fill

def analyze_student_sequence(student: str, sequence: list[str]) -> str:
    try:
        prompt = f"""
學生 {student} 的工程設計構面順序為：{' → '.join(sequence)}。
請針對該學生的學習歷程給予個別化建議，指出其構面是否合理、有無缺漏或改善建議。
請使用親切清楚的語氣，並以中文撰寫。
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ 無法產生 {student} 的建議：{str(e)}"

def generate_pdf(text: str = None, df: pd.DataFrame = None, mood_img: str = None, aspect_img: str = None, sequences: dict = None) -> str:
    print("開始生成 PDF")
    pdf = FPDF(format="A4")
    pdf.add_page()

    chinese_font_path = get_chinese_font_file()
    if not chinese_font_path:
        print("錯誤：無法取得中文字型檔，請先安裝合適的中文字型！")
        return ""

    pdf.add_font("ChineseFont", "", chinese_font_path, uni=True)
    pdf.add_font("ChineseFont", "B", chinese_font_path, uni=True)
    pdf.set_font("ChineseFont", "B", 20)
    pdf.cell(0, 15, "工程設計逐字稿分析報告", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("ChineseFont", "B", 16)
    pdf.cell(0, 10, "圖表分析", ln=True)
    pdf.ln(5)
    if mood_img and os.path.exists(mood_img):
        pdf.image(mood_img, x=10, w=180)
        pdf.ln(5)
    if aspect_img and os.path.exists(aspect_img):
        pdf.image(aspect_img, x=10, w=180)
        pdf.ln(10)

    pdf.add_page()
    pdf.set_font("ChineseFont", "B", 16)
    pdf.cell(0, 10, "全班建議分析", ln=True)
    pdf.ln(5)
    if text:
        pdf.set_font("ChineseFont", "", 12)
        formatted_lines = []
        for line in text.strip().splitlines():
            if "的構面順序：" in line and "→" not in line:
                parts = line.split("的構面順序：")
                student = parts[0].strip()
                phases = parts[1].strip().replace(",", " → ").replace("➜", " → ")
                formatted_line = f"{student} 的構面順序： {phases}"
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        full_text = "\n".join(formatted_lines)
        pdf.multi_cell(0, 10, full_text)
        pdf.ln(5)

    if sequences:
        pdf.add_page()
        pdf.set_font("ChineseFont", "B", 16)
        pdf.cell(0, 10, "個別學生建議分析", ln=True)
        pdf.ln(5)
        pdf.set_font("ChineseFont", "", 12)
        for student, seq in sequences.items():
            feedback = analyze_student_sequence(student, seq)
            pdf.set_font("ChineseFont", "B", 14)
            pdf.cell(0, 10, f"{student}：", ln=True)
            pdf.set_font("ChineseFont", "", 12)
            pdf.multi_cell(0, 10, feedback)
            pdf.ln(4)

    if df is not None:
        pdf.add_page()
        pdf.set_font("ChineseFont", "B", 16)
        pdf.cell(0, 10, "原始數據表格", ln=True)
        pdf.ln(5)
        create_table(pdf, df)

    output_dir = os.path.join(os.getcwd(), "static", "pdf")
    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = os.path.join(output_dir, "report.pdf")
    pdf.output(pdf_filename)
    print("PDF 生成完成：", pdf_filename)

    if not os.path.exists(pdf_filename):
        print("❌ 找不到 PDF 檔案：", pdf_filename)
        abort(404, description="PDF 檔案不存在")

    return pdf_filename
