# app.py

import os
import threading
import pandas as pd
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

from multiagent import run_multiagent_analysis

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, async_mode='threading')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

    # 啟動多代理分析
    sid = request.form.get("sid", "")  # 從前端 JS 傳入 socket ID
    thread = threading.Thread(target=run_multiagent_analysis, args=(filepath, socketio, sid))
    thread.start()

    return '檔案已上傳並開始分析', 200

if __name__ == '__main__':
    socketio.run(app, debug=True)
