# ✅ 修改後 app.py：加入個別學生建議分析並提供 /student_feedbacks 路由

import os
import threading
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from getPDF import generate_pdf, analyze_student_sequence
from multiagent import run_multiagent_analysis

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PDF_FOLDER'] = 'static/pdf'
socketio = SocketIO(app, async_mode='threading')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)

AI_FEEDBACK = ""
STUDENT_SEQUENCES = {}
STUDENT_FEEDBACKS = {}  # 🆕 每位學生建議文字
MOOD_IMG_PATH = ""
ASPECT_IMG_PATH = ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return '沒有檔案部分', 400
    file = request.files['file']
    if file.filename == '':
        return '沒有選擇檔案', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    sid = request.form.get("sid", "")
    thread = threading.Thread(target=run_multiagent_analysis, args=(filepath, socketio, sid))
    thread.start()

    return '檔案已上傳並開始分析', 200

@app.route('/set_feedback', methods=['POST'])
def set_feedback():
    global AI_FEEDBACK, STUDENT_SEQUENCES, MOOD_IMG_PATH, ASPECT_IMG_PATH, STUDENT_FEEDBACKS
    data = request.get_json()
    AI_FEEDBACK = data.get("ai_feedback", "")
    STUDENT_SEQUENCES = data.get("student_sequences", {})
    MOOD_IMG_PATH = data.get("moodtrend_img", "")
    ASPECT_IMG_PATH = data.get("aspect_bar_img", "")

    # 🔍 分析每位學生構面順序並產生建議
    STUDENT_FEEDBACKS = {}
    for student, seq in STUDENT_SEQUENCES.items():
        STUDENT_FEEDBACKS[student] = analyze_student_sequence(student, seq)

    return '接收成功', 200

@app.route('/student_feedbacks', methods=['GET'])
def student_feedbacks():
    return jsonify(STUDENT_FEEDBACKS)

@app.route('/generate_pdf', methods=['GET'])
def generate():
    mood_path = os.path.join(os.getcwd(), MOOD_IMG_PATH) if MOOD_IMG_PATH else None
    aspect_path = os.path.join(os.getcwd(), ASPECT_IMG_PATH) if ASPECT_IMG_PATH else None
    pdf_path = generate_pdf(
        text=AI_FEEDBACK,
        mood_img=mood_path,
        aspect_img=aspect_path,
        sequences=STUDENT_SEQUENCES
    )
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
