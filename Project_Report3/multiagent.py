# multiagent.py

import os
import json
import pandas as pd
import time
from flask_socketio import SocketIO
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
from snownlp import SnowNLP  # ✅ 新增情緒分析用

from EMOwithSnow import generate_analysis

# ✅ 設定 Gemini Flash 模型
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
configure(api_key=api_key)
gemini_model = GenerativeModel("gemini-1.5-flash")  # 若 gemini-2.0-flash 開放 API 名稱可改

def generate_gemini_feedback(summary_data):
    student_lines = [
        f"- {name} 的構面順序為：{' ➜ '.join(seq)}"
        for name, seq in summary_data["students"].items()
    ]
    joined_summary = "\n".join(student_lines)

    prompt = f"""
這是某班學生的工程設計逐字稿分析總結：
{joined_summary}

整體情緒平均值為：{summary_data["avg_sentiment"]:.2f}

請根據這些資訊，產出一段能協助班級改善學習歷程的建議，幫助學生補足設計過程中可能忽略的步驟。
建議請使用中文，並以親切、有建設性的語氣撰寫。
"""

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def run_multiagent_analysis(file_path, socketio: SocketIO, sid: str):
    try:
        socketio.emit("status", {"message": "🔍 開始讀取逐字稿資料..."}, to=sid)
        df = pd.read_csv(file_path)
        user_id = os.path.splitext(os.path.basename(file_path))[0]

        if "who" not in df.columns or "text" not in df.columns:
            socketio.emit("status", {"message": "❌ 錯誤：CSV 必須包含 'who' 和 'text' 欄位！"}, to=sid)
            return

        time.sleep(1)
        socketio.emit("status", {"message": "🤖 Agent 1：進行情緒分析中..."}, to=sid)

        time.sleep(1)
        socketio.emit("status", {"message": "🧠 Agent 2：工程構面分類中..."}, to=sid)

        # 執行完整分析（產圖＋分類）
        result_paths = generate_analysis(user_id, df)

        # ✅ 重新做一次簡單的情緒分析（只為了計算平均值）
        df["text"] = df["text"].fillna("").astype(str)
        df = df[df["text"].str.len() > 2].reset_index(drop=True)
        df["sentiment"] = df["text"].apply(lambda x: SnowNLP(x).sentiments)
        avg_sentiment = df["sentiment"].mean()

        # ✅ 讀入構面順序 JSON
        aspect_seq_path = result_paths["aspect_sequence_json"]
        if os.path.exists(aspect_seq_path):
            with open(aspect_seq_path, "r", encoding="utf-8") as f:
                student_sequences = json.load(f)
        else:
            student_sequences = {}

        # ✅ 傳入 Gemini 分析用摘要
        summary_data = {
            "avg_sentiment": avg_sentiment,
            "students": student_sequences
        }

        print("🧠 傳給 Gemini 的摘要內容如下：")
        print(json.dumps(summary_data, indent=2, ensure_ascii=False))

        socketio.emit("status", {"message": "✨ Gemini 正在生成 AI 建議中..."}, to=sid)

        feedback = generate_gemini_feedback(summary_data)

        print("✅ Gemini 產生的回應如下：")
        print(feedback)

        socketio.emit("status", {"message": "✅ 所有分析完成！正在生成視覺化結果..."}, to=sid)

        # 回傳所有資料給前端
        socketio.emit("result", {
            "moodtrend_img": result_paths["moodtrend_img"],
            "aspect_bar_img": result_paths["aspect_bar_img"],
            "aspect_sequence_json": result_paths["aspect_sequence_json"],
            "ai_feedback": feedback
        }, to=sid)

    except Exception as e:
        socketio.emit("status", {"message": f"❌ 發生錯誤：{str(e)}"}, to=sid)
