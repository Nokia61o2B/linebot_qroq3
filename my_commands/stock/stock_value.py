import numpy as np
import yfinance as yf
import re

# 基本面資料
def stock_fundamental(stock_id="大盤"):
    if stock_id == "大盤":
        return "大盤沒有具體基本面資料"

    # 判斷是否為台股（4-6位數字，可帶字母）
    if re.match(r'^\d{4,6}[A-Za-z]?$', stock_id):  # 台股代碼格式
        stock_id_tw = stock_id + ".TW"
        stock_id_two = stock_id + ".TWO"

        # 嘗試抓取 .TW 資料
        try:
            stock = yf.Ticker(stock_id_tw)
            quarterly_revenue_growth = np.round(
                stock.quarterly_financials.loc["Total Revenue"].pct_change(-1).dropna().tolist(), 2)
        except:
            # 若抓取 .TW 資料失敗，嘗試抓取 .TWO 資料
            try:
                stock = yf.Ticker(stock_id_two)
                quarterly_revenue_growth = np.round(
                    stock.quarterly_financials.loc["Total Revenue"].pct_change(-1).dropna().tolist(), 2)
            except Exception as e:
                return f"無法取得基本面資料: {e}"
    else:
        # 美股或其他市場的股票代碼
        try:
            stock = yf.Ticker(stock_id)
            quarterly_revenue_growth = np.round(
                stock.quarterly_financials.loc["Total Revenue"].pct_change(-1).dropna().tolist(), 2)
        except Exception as e:
            return f"無法取得基本面資料: {e}"

    # 檢查是否存在 "Reported EPS" 欄位
    if "Reported EPS" in stock.get_earnings_dates().columns:
        # 每季EPS
        quarterly_eps = np.round(
            stock.get_earnings_dates()["Reported EPS"].dropna().tolist(), 2)

        # EPS季增率
        quarterly_eps_growth = np.round(
            stock.get_earnings_dates()["Reported EPS"].ffill().pct_change(-1).dropna().tolist(), 2)
    else:
        quarterly_eps = [None] * 3
        quarterly_eps_growth = [None] * 3

    # 轉換日期
    dates = [
        date.strftime('%Y-%m-%d') for date in stock.quarterly_financials.columns
    ]

    data = {
        '季日期': dates[:len(quarterly_revenue_growth)],
        '營收成長率': quarterly_revenue_growth,
        'EPS': quarterly_eps[:3],
        'EPS 季增率': quarterly_eps_growth[:3]
    }

    return data

