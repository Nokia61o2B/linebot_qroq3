import numpy as np
import yfinance as yf
import re

# 配息資料
def stock_dividend(stock_id="大盤"):
    if stock_id == "大盤":
        return "大盤沒有具體配息資料"

    # 判斷是否為台股（4-6位數字，可帶字母）
    if re.match(r'^\d{4,6}[A-Za-z]?$', stock_id):  # 台股代碼格式
        stock_id_tw = stock_id + ".TW"
        stock_id_two = stock_id + ".TWO"

        # 嘗試抓取 .TW 資料
        try:
            stock = yf.Ticker(stock_id_tw)
            dividends = stock.dividends
        except:
            # 若抓取 .TW 資料失敗，嘗試抓取 .TWO 資料
            try:
                stock = yf.Ticker(stock_id_two)
                dividends = stock.dividends
            except Exception as e:
                return f"無法取得配息資料: {e}"
    else:
        # 美股或其他市場的股票代碼
        try:
            stock = yf.Ticker(stock_id)
            dividends = stock.dividends
        except Exception as e:
            return f"無法取得配息資料: {e}"

    # 檢查是否有配息資料
    if dividends.empty:
        return f"未找到 {stock_id} 的配息資料，請確認代號或資料源是否可用。"

    # 整理配息資料
    dates = [date.strftime('%Y-%m-%d') for date in dividends.index]  # 格式化日期
    dividend_values = np.round(dividends.values, 2).tolist()  # 取得配息金額

    # 將資料以字典形式返回
    data = {
        '日期': dates,
        '配息': dividend_values
    }

    return data