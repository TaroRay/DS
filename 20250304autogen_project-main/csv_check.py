import chardet

def check_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

# 例子：檢查 CSV 檔案的編碼
encoding = check_encoding('807.csv')
print(f"檔案編碼是: {encoding}")