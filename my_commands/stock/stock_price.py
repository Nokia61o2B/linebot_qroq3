import yfinance as yf
import datetime as dt
import re

# 從 yfinance 取得一周股價資料
def stock_price(stock_id, days=10):
    # 判斷是否為台股（4-6位數字，可帶字母）
    if re.match(r'^\d{4,6}[A-Za-z]?$', stock_id):  # 台股代碼格式
        stock_id_tw = stock_id + ".TW"
        stock_id_two = stock_id + ".TWO"

        end = dt.date.today()  # 資料結束時間
        start = end - dt.timedelta(days=days)  # 資料開始時間

        # 嘗試下載 .TW 資料
        try:
            df = yf.download(stock_id_tw, start=start, end=end)
            # 檢查是否有下載到資料
            if df.empty:
                raise Exception("No data found for .TW")
        except:
            # 若 .TW 資料下載失敗，嘗試下載 .TWO 資料
            try:
                df = yf.download(stock_id_two, start=start, end=end)
                if df.empty:
                    raise Exception("No data found for .TWO")
            except Exception as e:
                return f"無法下載股票資料: {e}"
    else:
        # 美股或其他市場的股票代碼
        end = dt.date.today()  # 資料結束時間
        start = end - dt.timedelta(days=days)  # 資料開始時間

        # 嘗試下載資料
        try:
            df = yf.download(stock_id, start=start, end=end)
            if df.empty:
                raise Exception("No data found for stock_id")
        except Exception as e:
            return f"無法下載股票資料: {e}"

    # 更換列名
    # df.columns = ['開盤價', '最高價', '最低價', '收盤價', '調整後收盤價', '成交量']
    df.columns = ['開盤價', '最高價', '最低價', '收盤價', '成交量']


    data = {
        '日期': df.index.strftime('%Y-%m-%d').tolist(),  # 格式化日期
        '收盤價': df['收盤價'].tolist(),  # 提取收盤價
        '每日報酬': df['收盤價'].pct_change().tolist(),  # 計算每日報酬率
        '成交量': df['成交量'].diff().tolist()  # 計算漲跌價差
    }

    return data

# 測試函數
# print(stock_price("2330", 10))  # 測試台股代碼
# print(stock_price("AAPL", 10))  # 測試美股代碼
