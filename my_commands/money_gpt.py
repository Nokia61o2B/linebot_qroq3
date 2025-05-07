# 使用其他方式抓取匯率數據
import os
import openai
from groq import Groq
import pandas as pd
import yfinance as yf  # 需要先 pip install yfinance

# 設定 API 金鑰
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 建立 GPT 模型
def get_reply(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages)
        reply = response["choices"][0]["message"]["content"]
    except openai.OpenAIError as openai_err:
        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=1000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except groq.GroqError as groq_err:
            reply = f"OpenAI API 發生錯誤: {openai_err.error.message}，GROQ API 發生錯誤: {groq_err.message}"
    return reply

# 用 yfinance 抓匯率數據的函式
def fetch_currency_rates(kind):
    """
    kind 參數請用 Yahoo Finance 代號，例如：
    'USDTWD=X' → 美元對台幣
    'TWDUSD=X' → 台幣對美元
    """
    ticker = yf.Ticker(kind)
    hist = ticker.history(period="30d")  # 取最近30天資料

    if hist.empty:
        raise ValueError(f"找不到代號 {kind} 的資料，請確認 Yahoo Finance 代碼是否正確。")

    # 重整 DataFrame
    df = hist[['Close']].reset_index()
    df.rename(columns={'Date': '日期', 'Close': '收盤價'}, inplace=True)
    df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')

    print(df)  # 調試輸出
    return df

# 建立分析報告內容
def generate_content_msg(kind):
    money_prices_df = fetch_currency_rates(kind)

    max_price = money_prices_df['收盤價'].max()
    min_price = money_prices_df['收盤價'].min()
    last_date = money_prices_df['日期'].iloc[-1]

    content_msg = f'你現在是一位專業的{kind}幣種分析師, 使用以下數據來撰寫分析報告:\n'
    content_msg += f'{money_prices_df.to_string(index=False)} 顯示最近30天資料,\n'
    content_msg += f'最新日期: {last_date}, 最高價: {max_price}, 最低價: {min_price}。\n'
    content_msg += '請給出完整的趨勢分析報告，顯示每日匯率 {日期} - {收盤價}(幣種/台幣)，'
    content_msg += '使用繁體中文。'

    return content_msg

# 主函式
def money_gpt(kind):
    content_msg = generate_content_msg(kind)
    print(content_msg)  # 調試輸出

    msg = [{
        "role": "system",
        "content": f"你現在是一位專業的{kind}幣種分析師, 使用以下數據來撰寫分析報告。"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_reply(msg)
    return reply_data