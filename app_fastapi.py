from fastapi import FastAPI, APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.models import PostbackEvent, TextSendMessage, MessageEvent, TextMessage
from linebot.models import *
from linebot.exceptions import LineBotApiError, InvalidSignatureError
import os
from openai import OpenAI
import requests
from groq import Groq
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from my_commands.stock.stock_gpt import stock_gpt
import re
import uvicorn
import asyncio
from contextlib import asynccontextmanager
import httpx
import os

app = FastAPI()
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        update_line_webhook()
    except Exception as e:
        print(f"更新 Webhook URL 失敗: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# SET BASE URL
base_url = os.getenv("BASE_URL")
# Channel Access Token
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
# Channel Secret
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
# 初始化 Groq API client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 初始化對話歷史
conversation_history = {}
MAX_HISTORY_LEN = 10  # 設定最大對話記憶長度

# 初始化 OpenAI 客戶端
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = "https://free.v36.cm/v1"
)

# 建立 GPT 模型
async def get_reply(messages):
    select_model = "gpt-4o-mini"
    print(f"free gpt:{select_model}")
    try:
        completion = await client.chat.completions.create(  # ✅ 這是 OpenAI 的 async client
            model=select_model,
            messages=messages,
            max_tokens=800  
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        print("OpenAI API 發生錯誤，嘗試使用 Groq:")
        try:
            # ❌ 不要 await，Groq 是同步的！
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                max_tokens=2000,
                temperature=1.2
            )
            reply = response.choices[0].message.content
        except Exception as groq_err:
            reply = f"OpenAI API 發生錯誤，GROQ API 發生錯誤: {str(groq_err)}"
    return reply

import requests  # 這邊保留 requests，不改成 aiohttp

# ✅ 改為同步版本
def check_line_webhook():
    url = "https://api.line.me/v2/bot/channel/webhook/endpoint"
    headers = {
        "Authorization": f"Bearer {os.getenv('CHANNEL_ACCESS_TOKEN')}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        current_webhook = response.json().get("endpoint", "無法取得 Webhook URL")
        print(f"當前 Webhook URL: {current_webhook}")
        return current_webhook
    else:
        print(f"檢查 Webhook URL 失敗，狀態碼: {response.status_code}, 原因: {response.text}")
        return None

# 定義 LINE Webhook 端點
LINE_WEBHOOK_ENDPOINT = "https://api.line.me/v2/bot/channel/webhook/endpoint"
def update_line_webhook():
    """同步更新 LINE Webhook URL"""
    access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    base_url = os.getenv("BASE_URL")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    json_data = {"endpoint": f"{base_url}/callback"}

    try:
        with httpx.Client() as client:
            res = client.put(LINE_WEBHOOK_ENDPOINT, headers=headers, json=json_data)
            res.raise_for_status()
            print(f"✅ Webhook 更新成功: {res.status_code}")
    except httpx.HTTPStatusError as e:
        print(f"❌ Webhook 更新失敗: {e.response.status_code} {e.response.text}")
    except Exception as e:
        print(f"❌ 發生未知錯誤: {e}")


router = APIRouter()

# ✅ 修正版 LINE Webhook 端點
@router.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature")

    try:
        # ✅ 呼叫同步的 handler.handle()（內部會觸發你註冊的 async 處理函式）
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"message": "ok"})
    
app.include_router(router)  # ⚠️ 一定要包含這行
# ✅ 修正版 LINE Webhook 處理訊息函式
# ✅ 解決 coroutine 'handle_message' was never awaited 錯誤
# ✅ 保留 async 用法，確保支援非同步 GPT API 呼叫與 LINE 回覆

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceGroup, SourceRoom
)
from linebot.exceptions import LineBotApiError
import re

@handler.add(MessageEvent, message=TextMessage)
def handle_message_wrapper(event):
    import asyncio
    asyncio.create_task(handle_message(event))  # ✅ 建立 background task

async def handle_message(event):
    global conversation_history
    user_id = event.source.user_id
    msg = event.message.text

    # 檢查是否為群組或聊天室訊息
    is_group_or_room = isinstance(event.source, (SourceGroup, SourceRoom))

    if is_group_or_room:
        bot_info = line_bot_api.get_bot_info()
        bot_name = bot_info.display_name

        if '@' in msg:
            at_text = msg.split('@')[1].split()[0] if len(msg.split('@')) > 1 else ''
            if at_text.lower() not in bot_name.lower():
                return
            msg = msg.replace(f'@{at_text}', '').strip()
        else:
            return

        if not msg:
            return

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    conversation_history[user_id].append({"role": "user", "content": msg + ", 請以繁體中文回答我問題"})

    stock_code = re.search(r'^\d{4,6}[A-Za-z]?\b', msg)
    stock_symbol = re.search(r'^[A-Za-z]{1,5}\b', msg)

    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN * 2:]

    lottery_keywords = ["威力彩", "大樂透", "539", "雙贏彩", "3星彩", "三星彩", "4星彩", "四星彩", "38樂合彩", "39樂合彩", "49樂合彩", "運彩"]

    try:
        if any(keyword in msg for keyword in lottery_keywords):
            reply_text = lottery_gpt(msg)
        elif msg.lower().startswith("大盤") or msg.lower().startswith("台股"):
            reply_text = stock_gpt("大盤")
        elif msg.lower().startswith("美盤") or msg.lower().startswith("美股"):
            reply_text = stock_gpt("美盤")
        elif stock_code:
            stock_id = stock_code.group()
            reply_text = stock_gpt(stock_id)
        elif stock_symbol:
            stock_id = stock_symbol.group()
            reply_text = stock_gpt(stock_id)
        elif any(msg.lower().startswith(currency.lower()) for currency in ["金價", "金", "黃金", "gold"]):
            reply_text = gold_gpt()
        elif any(msg.lower().startswith(currency.lower()) for currency in ["鉑", "鉑金", "platinum", "白金"]):
            reply_text = platinum_gpt()
        elif msg.lower().startswith(tuple(["日幣", "日元", "jpy", "換日幣"])):
            reply_text = money_gpt("JPY")
        elif any(msg.lower().startswith(currency.lower()) for currency in ["美金", "usd", "美元", "換美金"]):
            reply_text = money_gpt("USD")
        elif msg.startswith("104:"):
            reply_text = one04_gpt(msg[4:])
        elif msg.startswith("pt:"):
            reply_text = partjob_gpt(msg[3:])
        elif msg.startswith("cb:") or msg.startswith("$:"):
            coin_id = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
            reply_text = crypto_gpt(coin_id)
        else:
            messages = conversation_history[user_id][-MAX_HISTORY_LEN:]
            reply_text = await get_reply(messages)
    except Exception as e:
        reply_text = f"API 發生錯誤: {str(e)}"

    if not reply_text:
        reply_text = "抱歉，目前無法提供回應，請稍後再試。"

    try:
        line_bot_api.push_message(user_id, TextSendMessage(reply_text))  # ✅ 用 push_message 回覆
    except LineBotApiError as e:
        print(f"LINE 回覆失敗: {e}")

    conversation_history[user_id].append({"role": "assistant", "content": reply_text})


@handler.add(PostbackEvent)
async def handle_postback(event):
    print(event.postback.data)

#觀迎剛剛加入的人
@handler.add(MemberJoinedEvent)
async def welcome(event):
    uid = event.joined.members[0].user_id
    if isinstance(event.source, SourceGroup):
        gid = event.source.group_id
        profile = await line_bot_api.get_group_member_profile(gid, uid)
    elif isinstance(event.source, SourceRoom):
        rid = event.source.room_id
        profile = await line_bot_api.get_room_member_profile(rid, uid)
    else:
        profile = await line_bot_api.get_profile(uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name} 歡迎加入')
    await line_bot_api.reply_message(event.reply_token, message)

# ✅ Render 預設健康檢查路徑
@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

# ✅ 可選首頁 GET
@app.get("/")
async def root():
    return {"message": "Service is live."}
    

# 啟動應用
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    try:
        uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)
    except Exception as e:
        print(f"伺服器啟動失敗: {e}")