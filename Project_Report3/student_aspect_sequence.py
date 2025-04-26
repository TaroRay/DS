# student_aspect_sequence.py

import pandas as pd
import re
from collections import defaultdict

df = pd.read_csv("engineering_interview.csv")

keyword_map = {
    "問題定義": ["問題", "需求", "目的", "困難", "挑戰"],
    "背景調查": ["查資料", "搜尋", "了解", "研究", "問老師", "參考", "看影片"],
    "方案構想": ["想法", "構想", "創意", "討論", "計畫", "設想"],
    "方案設計": ["設計", "繪圖", "畫圖", "建模", "流程", "安排"],
    "作品製作": ["製作", "組裝", "材料", "黏", "剪", "焊", "做出來"],
    "測試與改善": ["測試", "試看看", "修改", "改善", "調整", "失敗", "重來"]
}

def classify_aspects(text):
    found = set()
    for aspect, keywords in keyword_map.items():
        for kw in keywords:
            if re.search(kw, text, flags=re.IGNORECASE):
                found.add(aspect)
    return list(found)

student_aspect_sequence = {}
for student, group in df.groupby("who"):
    sequence = []
    prev = None
    for content in group["text"]:
        for aspect in classify_aspects(content):
            if aspect != prev:
                sequence.append(aspect)
                prev = aspect
    student_aspect_sequence[student] = sequence

# 輸出結果
for student, seq in student_aspect_sequence.items():
    print(f"{student} 的構面順序：{' ➜ '.join(seq)}")
