# 蓉蓉小助理
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from linebot import LineBotApi, WebhookHandler
from linebot.models import *
from linebot.exceptions import LineBotApiError, InvalidSignatureError
import os, re, asyncio, httpx, uvicorn, requests
from contextlib import asynccontextmanager
from openai import OpenAI
from groq import Groq
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from my_commands.stock.stock_gpt import stock_gpt

# 建立 FastAPI 應用程式
app = FastAPI()

# 更新 LINE webhook
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        update_line_webhook()
    except Exception as e:
        print(f"更新 Webhook URL 失敗: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# 初始化
base_url = os.getenv("BASE_URL")
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
conversation_history = {}
MAX_HISTORY_LEN = 10
auto_reply_status = {}  # ✅ key: chat_id, value: True/False

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://free.v36.cm/v1")

def update_line_webhook():
    """更新 LINE webhook URL"""
    access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    json_data = {"endpoint": f"{base_url}/callback"}
    try:
        with httpx.Client() as client:
            res = client.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data)
            res.raise_for_status()
            print(f"✅ Webhook 更新成功: {res.status_code}")
    except Exception as e:
        print(f"❌ Webhook 更新失敗: {e}")

router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    """LINE Webhook callback 入口"""
    body = await request.body()
    signature = request.headers.get("X-Line-Signature")
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"message": "ok"})

app.include_router(router)

# ✅ 處理文字訊息（背景執行 → 用 push_message）
@handler.add(MessageEvent, message=TextMessage)
def handle_message_wrapper(event):
    asyncio.create_task(handle_message(event))  # → 保留背景執行

async def get_reply(messages):
    """使用 OpenAI / Groq 回覆訊息"""
    select_model = "gpt-4o-mini"
    try:
        completion = await client.chat.completions.create(model=select_model, messages=messages, max_tokens=800)
        return completion.choices[0].message.content
    except:
        try:
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192", messages=messages, max_tokens=2000, temperature=1.2
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI 發生錯誤: {str(e)}"

def show_loading_animation(user_id: str, seconds: int = 5):
    """觸發 LINE loading 動畫"""
    url = "https://api.line.me/v2/bot/chat/loading/start"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('CHANNEL_ACCESS_TOKEN')}"
    }
    data = {
        "chatId": user_id,
        "loadingSeconds": seconds
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 202:
            print("✅ 載入動畫觸發成功")
        else:
            print(f"❌ 載入動畫錯誤: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ 載入動畫請求失敗: {e}")

def calculate_english_ratio(text):
    """計算英文比例"""
    if not text:
        return 0
    english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    total_chars = sum(1 for c in text if c.isalpha())
    return english_chars / total_chars if total_chars > 0 else 0
    async def handle_message(event):
    """處理來自 LINE 的訊息（用 push_message，不用 reply_token）"""
    user_id = event.source.user_id
    msg = event.message.text
    is_group_or_room = isinstance(event.source, (SourceGroup, SourceRoom))

    chat_id = event.source.group_id if isinstance(event.source, SourceGroup) else (
              event.source.room_id if isinstance(event.source, SourceRoom) else user_id)

    if chat_id not in auto_reply_status:
        auto_reply_status[chat_id] = True

    if is_group_or_room:
        bot_info = line_bot_api.get_bot_info()
        bot_name = bot_info.display_name
        if not auto_reply_status[chat_id] and '@' not in msg:
            return

        if msg.strip().lower() == '開啟自動回答':
            line_bot_api.push_message(chat_id, TextSendMessage('已開啟自動回答功能'))
            auto_reply_status[chat_id] = True
            return
        elif msg.strip().lower() == '關閉自動回答':
            line_bot_api.push_message(chat_id, TextSendMessage('已關閉自動回答功能'))
            auto_reply_status[chat_id] = False
            return
    else:
        show_loading_animation(user_id)

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    conversation_history[user_id].append({"role": "user", "content": msg + ", 請以繁體中文回答我問題"})

    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN * 2:]

    reply_text = ""
    try:
        stock_code = re.search(r'^\d{4,6}[A-Za-z]?\b', msg)
        stock_symbol = re.search(r'^[A-Za-z]{1,5}\b', msg)

        if any(k in msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
            reply_text = lottery_gpt(msg)
        elif msg.lower().startswith(("大盤", "台股")):
            reply_text = stock_gpt("大盤")
        elif msg.lower().startswith(("美盤", "美股")):
            reply_text = stock_gpt("美盤")
        elif msg.startswith("pt:"):
            reply_text = partjob_gpt(msg[3:])
        elif any(msg.lower().startswith(k.lower()) for k in ["金價", "黃金", "gold"]):
            reply_text = gold_gpt()
        elif any(msg.lower().startswith(k.lower()) for k in ["鉑", "platinum"]):
            reply_text = platinum_gpt()
        elif any(msg.lower().startswith(k.lower()) for k in ["日幣", "jpy"]):
            reply_text = money_gpt("JPY")
        elif any(msg.lower().startswith(k.lower()) for k in ["美金", "usd"]):
            reply_text = money_gpt("USD")
        elif msg.startswith(("cb:", "$:")):
            coin_id = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
            reply_text = crypto_gpt(coin_id)
        elif stock_code:
            reply_text = stock_gpt(stock_code.group())
        elif stock_symbol:
            reply_text = stock_gpt(stock_symbol.group())
        elif msg.startswith("104:"):
            reply_text = one04_gpt(msg[4:])
        else:
            if auto_reply_status[chat_id]:
                reply_text = await get_reply(conversation_history[user_id][-MAX_HISTORY_LEN:])
            else:
                return
    except Exception as e:
        reply_text = f"API 發生錯誤: {str(e)}"

    if not reply_text:
        reply_text = "抱歉，目前無法提供回應，請稍後再試。"

    english_ratio = calculate_english_ratio(reply_text)
    has_high_english = english_ratio > 0.1

    quick_reply_items = []
    if has_high_english:
        quick_reply_items.append(QuickReplyButton(action=MessageAction(label="翻譯成中文", text="請將上述內容翻譯成繁體正體中文")))

    prefix = f"@{bot_name} " if is_group_or_room else ""
    quick_reply_items.extend([
        QuickReplyButton(action=MessageAction(label="開啟自動回答", text="開啟自動回答")),
        QuickReplyButton(action=MessageAction(label="關閉自動回答", text="關閉自動回答")),
        QuickReplyButton(action=MessageAction(label="台股大盤", text=f"{prefix}大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text=f"{prefix}美股")),
        QuickReplyButton(action=MessageAction(label="大樂透", text=f"{prefix}大樂透")),
        QuickReplyButton(action=MessageAction(label="威力彩", text=f"{prefix}威力彩")),
        QuickReplyButton(action=MessageAction(label="金價", text=f"{prefix}金價")),
        QuickReplyButton(action=MessageAction(label="日元", text=f"{prefix}JPY")),
        QuickReplyButton(action=MessageAction(label="美元", text=f"{prefix}USD")),
    ])

    reply_message = TextSendMessage(
        text=reply_text,
        quick_reply=QuickReply(items=quick_reply_items) if quick_reply_items else None
    )

    try:
        line_bot_api.push_message(user_id, reply_message)
        conversation_history[user_id].append({"role": "assistant", "content": reply_text})
    except LineBotApiError as e:
        print(f"❌ 發送訊息失敗: {e}")

@handler.add(PostbackEvent)
async def handle_postback(event):
    print(event.postback.data)

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
    message = TextSendMessage(text=f'{profile.display_name} 歡迎加入')
    await line_bot_api.push_message(uid, message)

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Service is live."}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)