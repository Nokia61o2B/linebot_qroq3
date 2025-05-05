"""
蓉蓉小助理（修正版）
✅ 修復 AsyncLineBotApi 初始化缺少 async_http_client 的問題
✅ FastAPI + AsyncLineBotApi + WebhookHandler + GPT 模組
✅ 全程 async 流程
✅ 繁體中文註解
"""

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import AsyncLineBotApi, WebhookHandler
from linebot.models import *
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.aiohttp_async_http_client import AioHttpClient  # ✅ 新增 import
import os, re, httpx, asyncio, uvicorn
from contextlib import asynccontextmanager
from openai import OpenAI
from groq import Groq

# ✅ 你的自訂指令模組
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from my_commands.stock.stock_gpt import stock_gpt

app = FastAPI()

# ✅ lifespan 啟動時更新 webhook
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await update_line_webhook()  # ✅ 改成 async
    except Exception as e:
        print(f"更新 Webhook URL 失敗: {e}")
    yield

app = FastAPI(lifespan=lifespan)

base_url = os.getenv("BASE_URL")

# ✅ 初始化 AsyncLineBotApi + AioHttpClient
async_http_client = AioHttpClient()  # ✅ 新增必要 client
line_bot_api = AsyncLineBotApi(channel_access_token=os.getenv('CHANNEL_ACCESS_TOKEN'), async_http_client=async_http_client)
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation_history = {}
MAX_HISTORY_LEN = 10
assistant_status = {}
group_assistant_status = False

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://free.v36.cm/v1")

async def update_line_webhook():
    """更新 LINE webhook endpoint"""
    access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    json_data = {"endpoint": f"{base_url}/callback"}
    async with httpx.AsyncClient() as client:
        res = await client.put("https://api.line.me/v2/bot/channel/webhook/endpoint",
                               headers=headers, json=json_data)
        res.raise_for_status()
        print(f"✅ Webhook 更新成功: {res.status_code}")

router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    """接收 LINE webhook 事件"""
    body = await request.body()
    signature = request.headers.get("X-Line-Signature")
    try:
        await handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"message": "ok"})

app.include_router(router)

@handler.add(MessageEvent, message=TextMessage)
async def handle_message(event):
    """處理 LINE 訊息事件"""
    global conversation_history, assistant_status, group_assistant_status
    user_id = event.source.user_id
    msg = event.message.text.strip()
    is_group_or_room = isinstance(event.source, (SourceGroup, SourceRoom))

    reply_text = "抱歉，目前無法提供回應，請稍後再試。"
    quick_reply_items = []

    if is_group_or_room:
        bot_info = await line_bot_api.get_bot_info()
        bot_name = bot_info.display_name

        if msg in ["助理應答[on]", "助理應答[off]"]:
            group_assistant_status = msg == "助理應答[on]"
            reply_text = f"群組助理應答已{'啟用' if group_assistant_status else '停用'}"
        else:
            if not group_assistant_status:
                if not msg.startswith('@'):
                    return
                at_text = msg.split('@')[1].split()[0] if len(msg.split('@')) > 1 else ''
                if at_text.lower() not in bot_name.lower():
                    return
                msg = msg.replace(f'@{at_text}', '').strip()
    else:
        bot_name = ""

    if is_group_or_room:
        quick_reply_items.append(
            QuickReplyButton(
                action=MessageAction(
                    label=f"助理應答[{'off' if group_assistant_status else 'on'}]",
                    text=f"助理應答[{'off' if group_assistant_status else 'on'}]"
                )
            )
        )

    prefix = f"@{bot_name} " if is_group_or_room else ""
    quick_reply_items.extend([
        QuickReplyButton(action=MessageAction(label="台股大盤", text=f"{prefix}大盤")),
        QuickReplyButton(action=MessageAction(label="美股大盤", text=f"{prefix}美股")),
        QuickReplyButton(action=MessageAction(label="大樂透", text=f"{prefix}大樂透")),
        QuickReplyButton(action=MessageAction(label="威力彩", text=f"{prefix}威力彩")),
        QuickReplyButton(action=MessageAction(label="金價", text=f"{prefix}金價")),
    ])

    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": "user", "content": msg + ", 請以繁體中文回答我問題"})

    stock_code = re.search(r'^\d{4,6}[A-Za-z]?\b', msg)
    stock_symbol = re.search(r'^[A-Za-z]{1,5}\b', msg)

    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN * 2:]

    try:
        if any(k in msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
            reply_text = lottery_gpt(msg)
        elif msg.lower().startswith("大盤") or msg.lower().startswith("台股"):
            reply_text = stock_gpt("大盤")
        elif msg.lower().startswith("美盤") or msg.lower().startswith("美股"):
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
        elif msg.startswith("cb:") or msg.startswith("$:"):
            coin_id = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
            reply_text = crypto_gpt(coin_id)
        elif stock_code:
            reply_text = stock_gpt(stock_code.group())
        elif stock_symbol:
            reply_text = stock_gpt(stock_symbol.group())
        elif msg.startswith("104:"):
            reply_text = one04_gpt(msg[4:])
        else:
            reply_text = await get_reply_async(conversation_history[user_id][-MAX_HISTORY_LEN:])
    except Exception as e:
        reply_text = f"API 發生錯誤：{str(e)}"

    english_ratio = calculate_english_ratio(reply_text)
    if english_ratio > 0.1:
        quick_reply_items.append(QuickReplyButton(action=MessageAction(label="翻譯成中文", text="請將上述內容翻譯成繁體中文")))

    try:
        if not reply_text.strip():
            reply_text = "抱歉，無法處理您的要求，請稍後再試。"
        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(
                text=reply_text,
                quick_reply=QuickReply(items=quick_reply_items) if quick_reply_items else None
            )
        )
        conversation_history[user_id].append({"role": "assistant", "content": reply_text})
    except LineBotApiError as e:
        print(f"❌ 回覆訊息失敗：{e}")
        if "Invalid reply token" in str(e):
            print("回覆權杖已過期，請及時回覆")

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
    await line_bot_api.reply_message(event.reply_token, message)

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Service is live."}

async def get_reply_async(messages):
    """async 版本 get_reply → 用 run_in_executor 包裝同步 OpenAI client"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    ).choices[0].message.content)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)