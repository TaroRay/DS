import os
import json
import time
import pandas as pd
import sys
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

# 載入 .env 中的 GEMINI_API_KEY
load_dotenv()

# 定義評分項目（依據原始 xlsx 編碼規則）
ITEMS = [
    "界定問題與條件限制",
    "蒐集資料限制",
    "發展方案",
    "預測分析",
    "選擇方案",
    "建模測試",
    "評估修正",
    "最佳化",
]

def parse_response(response_text):
    """
    嘗試解析 Gemini API 回傳的 JSON 格式結果。
    如果回傳內容被 markdown 的反引號包圍，則先移除這些標記。
    若解析失敗，則回傳所有項目皆為空的字典。
    """
    cleaned = response_text.strip()
    # 如果回傳內容以三個反引號開始，則移除第一行和最後一行
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    try:
        result = json.loads(cleaned)
        for item in ITEMS:
            if item not in result:
                result[item] = ""
        return result
    except Exception as e:
        print(f"解析 JSON 失敗：{e}")
        print("原始回傳內容：", response_text)
        return {item: "" for item in ITEMS}

def select_dialogue_column(chunk: pd.DataFrame) -> str:
    """
    根據 CSV 欄位內容自動選取存放逐字稿的欄位。
    優先檢查常見欄位名稱： "text", 
    若都不存在，則回傳第一個欄位。
    """
    preferred = ["text"]
    for col in preferred:
        if col in chunk.columns:
            return col
    print("CSV 欄位：", list(chunk.columns))
    return chunk.columns[0]

def process_batch_dialogue(client, dialogues: list, delimiter="-----"):
    """
    將多筆逐字稿合併成一個批次請求。
    提示中要求模型對每筆逐字稿產生 JSON 格式回覆，並以指定的 delimiter 分隔各筆結果。
    """
    prompt = (
        "你是一位工程設計的專家，請根據逐字稿內容評估該歸類為工程設計中：界定問題與條件限制、蒐集資料限制、發展方案、預測分析、選擇方案、建模測試、評估修正、最佳化及其他哪一個範疇\n"
        + "\n".join(ITEMS) +
        "\n\n請依據評估結果，對每個項目：若觸及則標記為 1，否則留空。"
        " 請對每筆逐字稿產生 JSON 格式回覆，並在各筆結果間用下列分隔線隔開：\n"
        f"{delimiter}\n"
        "例如：\n"
        "```json\n"
        "{\n  \"界定問題與條件限制\": \"1\",\n  \"蒐集資料限制\": \"\",\n  ...\n}\n"
        f"{delimiter}\n"
        "{{...}}\n"
    )
    batch_text = f"\n{delimiter}\n".join(dialogues)
    content = prompt + "\n\n" + batch_text

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content
        )
    except ServerError as e:
        print(f"API 呼叫失敗：{e}")
        return [{item: "" for item in ITEMS} for _ in dialogues]
    
    print("批次 API 回傳內容：", response.text)
    parts = response.text.split(delimiter)
    results = []
    for part in parts:
        part = part.strip()
        if part:
            results.append(parse_response(part))
    # 若結果數量多於原始筆數，僅取前面對應筆數；若不足則補足空結果
    if len(results) > len(dialogues):
        results = results[:len(dialogues)]
    elif len(results) < len(dialogues):
        results.extend([{item: "" for item in ITEMS}] * (len(dialogues) - len(results)))
    return results

def count_item_occurrences(batch_results):
    """
    統計每個項目出現的次數，若項目標記為 '1' 則累加該項目的出現次數
    """
    counts = {item: 0 for item in ITEMS}
    for result in batch_results:
        for item in ITEMS:
            if result.get(item, "") == "1":
                counts[item] += 1
    return counts

def main():
    input_csv = 'engineering_interview.csv'  # 直接指定你的 CSV 路徑
    output_csv = "engineer_output.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    df = pd.read_csv(input_csv)
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("請設定環境變數 GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)
    
    dialogue_col = select_dialogue_column(df)
    print(f"使用欄位作為逐字稿：{dialogue_col}")
    
    batch_size = 10
    total = len(df)
    total_counts = {item: 0 for item in ITEMS}
    
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = df.iloc[start_idx:end_idx]
        dialogues = batch[dialogue_col].tolist()
        dialogues = [str(d).strip() for d in dialogues]
        batch_results = process_batch_dialogue(client, dialogues)
        
        # 統計每個項目在此批次中的出現次數
        batch_counts = count_item_occurrences(batch_results)
        for item in ITEMS:
            total_counts[item] += batch_counts[item]
        
        # 將分析結果加入到原始的 batch 中
        batch_df = batch.copy()
        for item in ITEMS:
            batch_df[item] = [res.get(item, "") for res in batch_results]
        
        # 寫入到 CSV
        if start_idx == 0:
            batch_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        else:
            batch_df.to_csv(output_csv, mode='a', index=False, header=False, encoding="utf-8-sig")
        
        print(f"已處理 {end_idx} 筆 / {total}")
        time.sleep(1)
    
    # 在最後一行添加統計結果
    summary_df = pd.DataFrame([total_counts])
    summary_df.to_csv(output_csv, mode='a', header=False, index=False, encoding="utf-8-sig")
    print(f"全部處理完成。最終結果已寫入：{output_csv}")

if __name__ == "__main__":
    main()
