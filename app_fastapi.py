# app_fastapi.py
# 蓉蓉小助理：全程使用 reply_message，不佔用 push_message 額度
# github: git@nokia6102b.github.com:Nokia61o2B/linebot_qroq3.git

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.models import *
from linebot.exceptions import LineBotApiError, InvalidSignatureError
import os, re, asyncio, httpx
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

# ============================================
# FastAPI 生命週期
# ============================================
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        update_line_webhook()
    except Exception as e:
        print(f"更新 Webhook URL 失敗: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# ============================================
# 初始化 LINE、OpenAI、Groq
# ============================================
base_url = os.getenv("BASE_URL")
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 更新 Groq 模型使用最新可用的型号
GROQ_MODEL_PRIMARY = os.getenv("GROQ_MODEL_PRIMARY", "llama-3.3-70b-versatile")  # 主要模型
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")   # 备用模型
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# OpenAI（你目前走自定 base_url）
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://free.v36.cm/v1")

conversation_history = {}
MAX_HISTORY_LEN = 10
auto_reply_status = {}

# ============================================
# LINE Webhook 設定
# ============================================
def update_line_webhook():
    """更新 LINE webhook URL"""
    access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    json_data = {"endpoint": f"{base_url}/callback"}
    try:
        with httpx.Client() as client_http:
            res = client_http.put("https://api.line.me/v2/bot/channel/webhook/endpoint", headers=headers, json=json_data)
            res.raise_for_status()
            print(f"✅ Webhook 更新成功: {res.status_code}")
    except Exception as e:
        print(f"❌ Webhook 更新失敗: {e}")

router = APIRouter()

@router.post("/callback")
async def callback(request: Request):
    """LINE Webhook callback"""
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

# ============================================
# Groq 呼叫工具（主→備 自動切換）
# ============================================
def groq_chat_completion(messages, max_tokens=800, temperature=0.7):
    """统一的 Groq 聊天完成函数，支持主备模型切换"""
    try:
        # 首先尝试主要模型
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_PRIMARY,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content
    except Exception as e_primary:
        print(f"主要模型 {GROQ_MODEL_PRIMARY} 失敗: {e_primary}")
        try:
            # 主要模型失败时尝试备用模型
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL_FALLBACK,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return completion.choices[0].message.content
        except Exception as e_fallback:
            print(f"备用模型 {GROQ_MODEL_FALLBACK} 也失敗: {e_fallback}")
            return f"抱歉，AI 服務暫時不可用。錯誤信息: {str(e_fallback)}"

# ============================================
# 情緒分析
# ============================================
async def analyze_sentiment(text: str) -> str:
    """
    判斷訊息情緒
    回傳: positive / neutral / negative / angry
    """
    try:
        # 首先尝试 OpenAI
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一個情感分析助手，輸出文字情緒標籤"},
                {"role": "user", "content": f"判斷這句話的情緒：{text}\n只回傳一個標籤：positive, neutral, negative, angry"}
            ],
            max_tokens=10,
            temperature=0
        )
        return completion.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"OpenAI 情感分析失敗: {e}")
        # OpenAI 失敗時使用 Groq
        messages = [
            {"role": "system", "content": "你是一個情感分析助手，輸出文字情緒標籤"},
            {"role": "user", "content": f"判斷這句話的情緒：{text}\n只回傳一個標籤：positive, neutral, negative, angry"}
        ]
        result = groq_chat_completion(messages, max_tokens=10, temperature=0)
        return result.strip().lower() if result else "neutral"

# ============================================
# 一般聊天（帶入情緒標籤的 System Prompt）
# ============================================
async def get_reply(messages, sentiment: str = "neutral"):
    """用 OpenAI / Groq 回覆訊息（帶情緒提示）"""
    system_prompt = f"""
你是溫柔的 AI 女友，要根據使用者的情緒調整語氣。
情緒標籤：{sentiment}
- positive: 活潑興奮，跟著開心。
- negative: 安慰、貼心。
- angry: 安撫、冷靜。
- neutral: 正常聊天。
回覆請用繁體中文，保持女友般的口吻與自然節奏。
""".strip()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        # 首先尝试 OpenAI
        completion = await client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=full_messages, 
            max_tokens=800, 
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI 回覆失敗: {e}")
        # OpenAI 失敗時使用 Groq
        return groq_chat_completion(full_messages, max_tokens=800, temperature=0.7)

# ============================================
# 其他工具
# ============================================
def calculate_english_ratio(text):
    if not text:
        return 0
    english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    total_chars = sum(1 for c in text if c.isalpha())
    return english_chars / total_chars if total_chars > 0 else 0

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    asyncio.create_task(handle_message_async(event))

# ============================================
# 主處理流程
# ============================================
async def handle_message_async(event):
    user_id = event.source.user_id
    msg = event.message.text
    reply_token = event.reply_token
    is_group_or_room = isinstance(event.source, (SourceGroup, SourceRoom))

    chat_id = event.source.group_id if isinstance(event.source, SourceGroup) else (
        event.source.room_id if isinstance(event.source, SourceRoom) else user_id
    )

    if chat_id not in auto_reply_status:
        # 單人聊天預設開啟，群組聊天預設關閉
        auto_reply_status[chat_id] = not is_group_or_room

    bot_info = line_bot_api.get_bot_info()
    bot_name = bot_info.display_name

    if msg.strip().lower() == '開啟自動回答':
        auto_reply_status[chat_id] = True
        await reply_simple(reply_token, "✅ 已開啟自動回答")
        return
    elif msg.strip().lower() == '關閉自動回答':
        auto_reply_status[chat_id] = False
        await reply_simple(reply_token, "✅ 已關閉自動回答")
        return

    if not auto_reply_status[chat_id]:
        # 非自動回答模式下，檢查訊息是否包含機器人名稱
        if not any(name in msg.lower() for name in bot_name.lower().split()):
            return
        msg_parts = msg.split(None, 1)
        if len(msg_parts) > 1:
            msg = msg_parts[1]
        else:
            auto_reply_status[chat_id] = True
            await reply_simple(reply_token, "✅ 已開啟自動回答")
            return

    # 解析股票/代幣
    stock_code = re.search(r'^\d{4,6}[A-Za-z]?\b$', msg)
    stock_symbol = re.search(r'^[A-Za-z]{1,5}\b$', msg)

    if user_id not in conversation_history:
        conversation_history[user_id] = []
    conversation_history[user_id].append({"role": "user", "content": msg + ", 請以繁體中文回答我問題"})
    if len(conversation_history[user_id]) > MAX_HISTORY_LEN * 2:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY_LEN * 2:]

    reply_text = ""
    try:
        if any(k in msg for k in ["威力彩", "大樂透", "539", "雙贏彩"]):
            reply_text = lottery_gpt(msg)
        elif msg.lower().endswith(("大盤", "台股")):
            reply_text = stock_gpt("大盤")
        elif msg.lower().endswith(("美盤", "美股")):
            reply_text = stock_gpt("美盤")
        elif msg.startswith("pt:"):
            reply_text = partjob_gpt(msg[3:])
        elif any(msg.lower().endswith(k) for k in ["金價", "黃金", "gold"]):
            reply_text = gold_gpt()
        elif any(msg.lower().endswith(k) for k in ["鉑", "platinum"]):
            reply_text = platinum_gpt()
        elif any(msg.upper().endswith(k) for k in ["日幣", "JPY"]):
            reply_text = money_gpt(f"{msg}TWD=X")
        elif any(msg.upper().endswith(k) for k in ["美金", "USD"]):
            reply_text = money_gpt(f"{msg}TWD=X")
        elif msg.startswith(("cb:", "$:")):
            coin_id = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
            reply_text = crypto_gpt(coin_id)
        elif stock_code:
            reply_text = stock_gpt(stock_code.group())
        elif stock_symbol:
            reply_text = stock_gpt(stock_symbol.group())
        else:
            # 情感分析 → 帶情緒產生回覆
            sentiment = await analyze_sentiment(msg)
            reply_text = await get_reply(conversation_history[user_id][-MAX_HISTORY_LEN:], sentiment=sentiment)
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
        QuickReplyButton(action=MessageAction(label="蓉蓉", text="@蓉蓉")),
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
        line_bot_api.reply_message(reply_token, reply_message)
        conversation_history[user_id].append({"role": "assistant", "content": reply_text})
    except LineBotApiError as e:
        print(f"❌ 回覆訊息失敗: {e}")

# ============================================
# 健康檢查
# ============================================
async def reply_simple(reply_token, text):
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=text))
    except LineBotApiError as e:
        print(f"❌ 回覆訊息失敗: {e}")

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Service is live."}