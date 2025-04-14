import os
import openai
from groq import Groq
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

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

def fetch_jpy_rates(kind):
    # 目標網址
    url = f"https://rate.bot.com.tw/xrt/quote/day/{kind}"

    # 發送HTTP請求
    response = requests.get(url)

    # 確定HTTP請求成功
    if response.status_code == 200:
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 在網頁中找到表格並提取即期匯率和本行賣出價格
        rows = soup.select("table.table-striped tbody tr")

        time_rates = []  # Initialize time_rates list
        spot_rates = []
        selling_rates = []

        for row in rows:
            columns = row.find_all('td')
            # 目前網頁結構下，即期匯率位於第3列，本行賣出價格位於第4列
            time_rate = columns[0].text
            spot_rate = columns[2].text
            selling_rate = columns[3].text

            print(f"日期:{time_rate} 即期匯率: {spot_rate}, 本行賣出價格: {selling_rate}")
            # 将即期匯率和本行賣出價格添加到列表中
            time_rates.append(time_rate)
            spot_rates.append(spot_rate)
            selling_rates.append(selling_rate)

        # 创建 Pandas DataFrame
        df = pd.DataFrame({
            '日期': time_rates,  # Renamed to match the data
            '即期匯率': spot_rates,
            '本行賣出價格': selling_rates
        })

        return df

def generate_content_msg(kind):
    # 获取和处理数据
    money_prices_df = fetch_jpy_rates(kind)

    # 从数据中获取需要的最高价和最低价信息
    max_price = money_prices_df['本行賣出價格'].max()  # 最高本行賣出價格
    min_price = money_prices_df['本行賣出價格'].min()  # 最低本行賣出價格
    last_date = money_prices_df['日期'].iloc[-1]  # 假设最后一行是最新日期数据

    # 构造专业分析报告的内容
    content_msg = f'你現在是一位專業的{kind}幣種分析師, 使用以下数据来撰写分析报告:\n'
    content_msg += f'{money_prices_df} 顯示最近的30筆,\n'
    content_msg += f'最新日期: {last_date}, 最高價: {max_price} {{日期}}-{{時間}}, 最低價: {min_price}{{日期}}-{{時間}}。\n'
    content_msg += '請給出完整的趨勢分析報告，顯示每日匯率{日期-時間}{匯率}(幣種/台幣)，'
    content_msg += '使用繁體中文。'

    return content_msg

def money_gpt(kind):
    content_msg = generate_content_msg(kind)
    print(content_msg)  # 调试输出

    msg = [{
        "role": "system",
        "content": f"你現在是一位專業的{kind}幣種分析師, 使用以下数据来撰写分析报告。"
    }, {
        "role": "user",
        "content": content_msg
    }]

    reply_data = get_reply(msg)
    return reply_data
