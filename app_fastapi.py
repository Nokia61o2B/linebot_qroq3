"""
蓉蓉小助理（最終修正版 v2）
✅ 採用同步版 LineBotApi，移除所有 AioHttpClient / AsyncLineBotApi
✅ 在 async handler 中以 run_in_executor 包裝同步 HTTP 呼叫
✅ FastAPI + WebhookHandler + GPT 模組
✅ 全程 async 流程（callback、startup）／sync 呼叫（reply_message、get_profile…）
✅ 繁體中文詳細註解
"""

import os
import re
import asyncio
import uvicorn
import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# LINE Bot 同步 SDK
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage, QuickReply, QuickReplyButton, MessageAction
from linebot.exceptions import InvalidSignatureError, LineBotApiError

# OpenAI / GROQ
from openai import OpenAI
from groq import Groq

# 自訂指令模組
from my_commands.lottery_gpt import lottery_gpt
from my_commands.gold_gpt import gold_gpt
from my_commands.platinum_gpt import platinum_gpt
from my_commands.money_gpt import money_gpt
from my_commands.one04_gpt import one04_gpt
from my_commands.partjob_gpt import partjob_gpt
from my_commands.crypto_coin_gpt import crypto_gpt
from my_commands.stock.stock_gpt import stock_gpt

# ----- 全域設定 -----
BASE_URL = os.getenv("BASE_URL")
CHANNEL_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_TOKEN)            # 同步版
handler      = WebhookHandler(CHANNEL_SECRET)       # 同步版
groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client= OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://free.v36.cm/v1")

conversation_history = {}
MAX_HISTORY_LEN    = 10
group_assistant_on = False  # 群組共用開關

# ----- FastAPI App & Lifespan -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時更新 Webhook endpoint
    try:
        await update_line_webhook()
    except Exception as e:
        print(f"Webhook 更新失敗: {e}")
    yield

app    = FastAPI(lifespan=lifespan)
router = APIRouter()

# ----- 更新 LINE Webhook Endpoint -----
async def update_line_webhook():
    """將 Webhook URL 設為 /callback"""
    headers = {
        "Authorization": f"Bearer {CHANNEL_TOKEN}",
        "Content-Type": "application/json"
    }
    json_data = {"endpoint": f"{BASE_URL}/callback"}
    async with httpx.AsyncClient() as client:
        res = await client.put(
            "https://api.line.me/v2/bot/channel/webhook/endpoint",
            headers=headers, json=json_data
        )
        res.raise_for_status()
        print(f"✅ Webhook 更新成功: {res.status_code}")

# ----- Callback Route -----
@router.post("/callback")
async def callback(request: Request):
    body      = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        # 同步 handler
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"message": "成功"})

app.include_router(router)

# ----- 工具函式 -----
def calculate_english_ratio(text: str) -> float:
    """計算字串中英文字母比例"""
    eng = re.findall(r"[A-Za-z]", text)
    return len(eng) / len(text) if text else 0.0

async def get_reply_async(messages):
    """在執行緒中呼叫 OpenAI/GROQ，同步包 async"""
    loop = asyncio.get_event_loop()
    def sync_call():
        # 優先 OpenAI
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-3.5-turbo-1106", messages=messages
            )
            return resp.choices[0].message.content
        except Exception as oe:
            # fallback 到 GROQ
            try:
                groq_resp = groq_client.chat.completions.create(
                    model="llama3-70b-8192", messages=messages,
                    max_tokens=1000, temperature=1.2
                )
                return groq_resp.choices[0].message.content
            except Exception as ge:
                return f"OpenAI 錯誤：{oe}，GROQ 錯誤：{ge}"
    return await loop.run_in_executor(None, sync_call)

# ----- 處理 MessageEvent -----
@handler.add("message", message="text")
async def handle_message(event):
    global conversation_history, group_assistant_on

    user_id = event.source.user_id
    msg     = event.message.text.strip()
    is_group= isinstance(event.source, (type(event.source).__bases__[0],))  # 判斷群組/房間（簡化）

    reply_text = "抱歉，無法處理，請稍後再試。"
    quick_rep  = []

    # ==== 群組 @ 助理 開關邏輯 ====
    if is_group:
        profile = line_bot_api.get_profile(user_id)
        bot_name= profile.display_name

        if msg in ["助理應答[on]", "助理應答[off]"]:
            group_assistant_on = (msg=="助理應答[on]")
            reply_text = f"群組助理應答已{'啟用' if group_assistant_on else '停用'}"
        else:
            if not group_assistant_on:
                if not msg.startswith(f"@{bot_name}"):
                    return  # 不回應
                # 去掉 @botName
                msg = msg.split(None,1)[1] if " " in msg else ""

        # 加入開關按鈕
        quick_rep.append(
            QuickReplyButton(
                action=MessageAction(
                    label=f"助理應答[{'off' if group_assistant_on else 'on'}]",
                    text=f"助理應答[{'off' if group_assistant_on else 'on'}]"
                )
            )
        )

    # ==== 常用快速按鈕 ====
    for label, cmd in [("台股大盤","大盤"),("美股大盤","美股"),("大樂透","大樂透"),
                       ("威力彩","威力彩"),("金價","金價")]:
        quick_rep.append(
            QuickReplyButton(
                action=MessageAction(
                    label=label, text=cmd
                )
            )
        )

    # ==== 記錄對話歷史 ====
    conversation_history.setdefault(user_id, [])
    conversation_history[user_id].append({"role":"user","content":msg+"，請以繁體中文回答"})
    if len(conversation_history[user_id])>MAX_HISTORY_LEN*2:
        conversation_history[user_id]=conversation_history[user_id][-MAX_HISTORY_LEN*2:]

    # ==== 指令判別 ====
    try:
        low = msg.lower()
        if any(k in msg for k in ["威力彩","大樂透","539","雙贏彩"]):
            reply_text = lottery_gpt(msg)
        elif low.startswith(("大盤","台股")):
            reply_text = stock_speech := stock_gpt("大盤")
        elif low.startswith(("美盤","美股")):
            reply_text = stock_gpt("美盤")
        elif msg.startswith("pt:"):
            reply_text = partjob_gpt(msg[3:])
        elif any(low.startswith(k) for k in ["金價","黃金","gold"]):
            reply_text = gold_gpt()
        elif any(low.startswith(k) for k in ["鉑","platinum"]):
            reply_text = platinum_gpt()
        elif any(low.startswith(k) for k in ["日幣","jpy"]):
            reply_text = money_gpt("JPY")
        elif any(low.startswith(k) for k in ["美金","usd"]):
            reply_text = money_gpt("USD")
        elif msg.startswith(("cb:","$:")):
            cid = msg[3:].strip() if msg.startswith("cb:") else msg[2:].strip()
            reply_text = crypto_gpt(cid)
        elif re.fullmatch(r"\d{4,6}[A-Za-z]?", msg):
            reply_text = stock_gpt(msg)
        elif re.fullmatch(r"[A-Za-z]{1,5}", msg):
            reply_text = stock_gpt(msg)
        elif msg.startswith("104:"):
            reply_text = one04_gpt(msg[4:])
        else:
            # ChatGPT/GROQ fallback
            reply_text = await get_reply_async(conversation_history[user_id][-MAX_HISTORY_LEN:])
    except Exception as e:
        reply_text = f"API 發生錯誤：{e}"

    # 若英文比例 > 10%，加翻譯按鈕
    if calculate_english_ratio(reply_text) > 0.1:
        quick_rep.append(
            QuickReplyButton(
                action=MessageAction(
                    label="翻譯成中文", text="請將上述內容翻譯成繁體中文"
                )
            )
        )

    # ==== 回覆 LINE ====
    # 包同步呼叫於執行緒，避免阻塞 event loop
    async def _reply():
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text,
                            quick_reply=QuickReply(items=quick_rep) if quick_rep else None)
        )
    await asyncio.get_event_loop().run_in_executor(None, _reply)

    # 紀錄助理回覆
    conversation_history[user_id].append({"role":"assistant","content":reply_text})

# ----- 處理其他事件 -----
@handler.add("postback")
def handle_postback(event):
    print(event.postback.data)

@handler.add("member_joined")
async def welcome(event):
    uid = event.joined.members[0].user_id
    profile = line_bot_api.get_profile(uid)
    # 同理包執行緒
    async def _welcome():
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"{profile.display_name} 歡迎加入！")
        )
    await asyncio.get_event_loop().run_in_executor(None, _welcome)

# ----- 健康檢查 & Root -----
@app.get("/healthz")
async def healthz():
    return {"status":"ok"}

@app.get("/")
async def root():
    return {"message":"服務正常運作中"}

# ----- 啟動 -----
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)