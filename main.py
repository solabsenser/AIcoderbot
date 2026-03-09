import asyncio
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime, timezone

import aiosqlite
import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    LabeledPrice,
    MenuButtonWebApp,
    PreCheckoutQuery,
    Update,
    WebAppInfo,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ======================= CONFIG =======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
APP_URL = os.getenv("APP_URL", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")
MINIAPP_URL = os.getenv("MINIAPP_URL", APP_URL)
ADMIN_IDS = {
    int(item.strip())
    for item in os.getenv("ADMIN_IDS", "").split(",")
    if item.strip().isdigit()
}
STAR_DONATE_AMOUNT = int(os.getenv("STAR_DONATE_AMOUNT", "50"))

DB_PATH = "db.sqlite"

SYSTEM_PROMPT = """
You are an elite senior Python developer.
Generate clean, production-ready Python 3.11 code.
Return ONLY full working code.
"""


# ======================= APP + BOT =======================
app = FastAPI()
session = AiohttpSession(timeout=60)
bot = Bot(token=BOT_TOKEN, session=session)
dp = Dispatcher()
APP_STARTED_AT = datetime.now(timezone.utc)


# ======================= MODELS ======================
class GenerateRequest(BaseModel):
    user_id: int = 0
    text: str


class SaveProjectRequest(BaseModel):
    user_id: int
    title: str
    task: str
    code: str


class DeleteProjectRequest(BaseModel):
    user_id: int
    project_id: int


class SendProjectRequest(BaseModel):
    user_id: int
    title: str
    code: str


class TestRequest(BaseModel):
    code: str


class AdminViewEventRequest(BaseModel):
    user_id: int = Field(..., gt=0)


# ======================= DATABASE =====================
async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                task TEXT,
                code TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.commit()


async def track_event(user_id: int, event_name: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO analytics_events (user_id, event_name)
            VALUES (?, ?)
            """,
            (user_id, event_name),
        )
        await db.commit()


async def get_admin_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        totals = await db.execute_fetchone(
            """
            SELECT
              COUNT(DISTINCT user_id) as users,
              COUNT(*) as projects
            FROM projects
            """
        )
        events = await db.execute_fetchall(
            """
            SELECT event_name, COUNT(*)
            FROM analytics_events
            GROUP BY event_name
            ORDER BY COUNT(*) DESC
            """
        )

    return {
        "users": totals[0] if totals else 0,
        "projects": totals[1] if totals else 0,
        "events": [{"event": name, "count": count} for name, count in events],
        "uptime_seconds": int((datetime.now(timezone.utc) - APP_STARTED_AT).total_seconds()),
    }


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


# ======================= LLM ==========================
async def call_llm(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()["choices"][0]["message"]["content"]


# ======================= API ==========================
@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        code = await call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.text},
            ]
        )
        if req.user_id > 0:
            await track_event(req.user_id, "generate")
        return {"code": code or ""}
    except Exception as error:
        print("❌ GENERATE ERROR:", repr(error))
        return {"error": str(error)}


@app.post("/projects/save")
async def save_project(payload: SaveProjectRequest):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO projects (user_id, title, task, code)
                VALUES (?, ?, ?, ?)
                """,
                (payload.user_id, payload.title, payload.task, payload.code),
            )
            await db.commit()
        await track_event(payload.user_id, "save")
        return {"status": "ok"}
    except Exception as error:
        print("❌ SAVE PROJECT ERROR:", repr(error))
        return {"error": "Failed to save project"}


@app.post("/projects/send_to_chat")
async def send_project_to_chat(payload: SendProjectRequest):
    if payload.user_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id. Open Mini App from Telegram bot.",
        )

    async def _send():
        try:
            document = BufferedInputFile(
                payload.code.encode("utf-8"),
                filename=f"{payload.title or 'project'}.py",
            )
            await bot.send_document(
                chat_id=payload.user_id,
                document=document,
                caption=f"📦 {payload.title}",
            )
            await track_event(payload.user_id, "send_to_chat")
        except TelegramBadRequest as error:
            print(f"⚠️ Telegram error for user {payload.user_id}: {error}")
        except Exception as error:
            print(f"❌ send_to_chat error: {error}")

    asyncio.create_task(_send())
    return {"status": "queued"}


@app.get("/projects/list/{user_id}")
async def list_projects(user_id: int):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            rows = await db.execute_fetchall(
                """
                SELECT id, title
                FROM projects
                WHERE user_id=?
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
        return [{"id": row[0], "title": row[1]} for row in rows]
    except Exception as error:
        print("❌ LIST PROJECTS ERROR:", repr(error))
        return []


@app.get("/projects/{project_id}")
async def get_project(project_id: int):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            row = await db.execute_fetchone(
                """
                SELECT title, task, code
                FROM projects
                WHERE id=?
                """,
                (project_id,),
            )

        if not row:
            return {"error": "Project not found"}

        return {"title": row[0], "task": row[1], "code": row[2]}
    except Exception as error:
        print("❌ GET PROJECT ERROR:", repr(error))
        return {"error": "Failed to load project"}


@app.post("/projects/delete")
async def delete_project(payload: DeleteProjectRequest):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                DELETE FROM projects
                WHERE id=? AND user_id=?
                """,
                (payload.project_id, payload.user_id),
            )
            await db.commit()
        await track_event(payload.user_id, "delete")
        return {"status": "deleted"}
    except Exception as error:
        print("❌ DELETE PROJECT ERROR:", repr(error))
        return {"error": "Failed to delete project"}


@app.post("/tests/run")
async def run_tests(req: TestRequest):
    with tempfile.TemporaryDirectory() as tmp:
        app_file = os.path.join(tmp, "app.py")
        test_file = os.path.join(tmp, "test_app.py")

        with open(app_file, "w", encoding="utf-8") as app_handle:
            app_handle.write(req.code)

        with open(test_file, "w", encoding="utf-8") as test_handle:
            test_handle.write(
                textwrap.dedent(
                    """
                    import app

                    def test_import():
                        assert app is not None
                    """
                )
            )

        try:
            result = subprocess.run(
                ["pytest", "-q"],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {"ok": result.returncode == 0, "output": result.stdout + result.stderr}
        except Exception as error:
            return {"ok": False, "output": str(error)}


@app.post("/admin/track_view")
async def admin_track_view(payload: AdminViewEventRequest):
    await track_event(payload.user_id, "admin_panel_open")
    return {"status": "ok"}


@app.get("/admin/stats/{user_id}")
async def admin_stats(user_id: int):
    if not is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin access required")
    return await get_admin_stats()


# ======================= MINI APP =====================
@app.get("/", response_class=HTMLResponse)
async def mini_app():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
<title>AI Code Studio Pro</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
:root {
  --bg: #060a12;
  --card: #111827;
  --border: #253046;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --accent: #6d7dff;
  --accent2: #22c55e;
  --danger: #ef4444;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at top, #111827, #050810 70%);
  color: var(--text);
  font-family: Inter, system-ui, sans-serif;
}
.container { padding: 14px; display: grid; gap: 14px; }
.card {
  background: rgba(15, 23, 42, 0.85);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}
.h1 { font-size: 16px; font-weight: 700; margin-bottom: 8px; }
.hint { color: var(--muted); font-size: 13px; margin-bottom: 10px; }
textarea, select, pre {
  width: 100%;
  border: 1px solid #1f2937;
  background: #020617;
  color: var(--text);
  border-radius: 12px;
  padding: 12px;
}
textarea { min-height: 130px; resize: vertical; }
pre { min-height: 180px; white-space: pre-wrap; overflow-wrap: anywhere; }
button {
  border: none;
  border-radius: 12px;
  padding: 12px;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
}
.btn-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 8px; }
.primary { background: linear-gradient(90deg, var(--accent), #9ca3ff); color: #fff; }
.success { background: linear-gradient(90deg, var(--accent2), #4ade80); color: #022c22; }
.danger { background: linear-gradient(90deg, var(--danger), #fb7185); color: #fff; }
.muted { background: #1f2937; color: #d1d5db; }
.badge {
  display: inline-block;
  background: #1e293b;
  border: 1px solid #334155;
  color: #93c5fd;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
}
#adminCard { display: none; }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <div class="h1">🚀 AI Code Studio Pro</div>
    <div class="hint">Быстрее генерация, лучше структура и встроенная аналитика для админов.</div>
    <span class="badge" id="roleBadge">User mode</span>
  </div>

  <div class="card">
    <div class="h1">📁 Проекты</div>
    <div class="hint">Выберите проект или начните новый.</div>
    <select id="projectSelect"></select>
  </div>

  <div class="card">
    <div class="h1">✍️ Задача</div>
    <textarea id="taskText" placeholder="Например: FastAPI + JWT + Redis + Docker Compose"></textarea>
  </div>

  <div class="card">
    <div class="h1">💻 Сгенерированный код</div>
    <pre id="codeText">// ожидание генерации...</pre>
  </div>

  <div class="card">
    <button class="primary" id="btnGenerate">⚡ Сгенерировать</button>
    <div class="btn-row">
      <button class="success" id="btnSave">💾 Сохранить</button>
      <button class="danger" id="btnDelete">🗑 Удалить</button>
      <button class="primary" id="btnSend">📤 В чат</button>
      <button class="muted" id="btnDonate">⭐ Поддержать</button>
    </div>
  </div>

  <div class="card" id="adminCard">
    <div class="h1">🛡 Admin Panel</div>
    <div class="hint">Статистика по использованию бота.</div>
    <pre id="adminStats">loading...</pre>
  </div>
</div>

<script>
let USER_ID = null;
let IS_ADMIN = false;
let currentProject = null;
const API = location.origin;

const select = document.getElementById('projectSelect');
const taskText = document.getElementById('taskText');
const codeText = document.getElementById('codeText');
const adminCard = document.getElementById('adminCard');
const adminStats = document.getElementById('adminStats');
const roleBadge = document.getElementById('roleBadge');

if (window.Telegram && window.Telegram.WebApp) {
  const tg = window.Telegram.WebApp;
  tg.expand(); tg.ready();
  USER_ID = tg.initDataUnsafe?.user?.id || null;
}

function toast(text) { alert(text); }

function renderEmptySelect() {
  select.innerHTML = '<option value="">➕ New project</option>';
}

async function loadProjects() {
  renderEmptySelect();
  if (!USER_ID) return;
  const r = await fetch(API + '/projects/list/' + USER_ID);
  const data = await r.json();
  select.innerHTML = '<option value="">➕ New project</option>' + data.map(p => `<option value="${p.id}">${p.title}</option>`).join('');
}

select.onchange = async () => {
  if (!select.value) {
    currentProject = null;
    taskText.value = '';
    codeText.textContent = '';
    return;
  }
  currentProject = select.value;
  const r = await fetch(API + '/projects/' + currentProject);
  const p = await r.json();
  taskText.value = p.task || '';
  codeText.textContent = p.code || '';
};

document.getElementById('btnGenerate').onclick = async () => {
  codeText.textContent = '⏳ Генерация...';
  const r = await fetch(API + '/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: USER_ID || 0, text: taskText.value })
  });
  const data = await r.json();
  codeText.textContent = data.code || ('❌ ' + (data.error || 'Ошибка генерации'));
};

document.getElementById('btnSave').onclick = async () => {
  if (!USER_ID) return toast('Откройте Mini App из Telegram бота.');
  await fetch(API + '/projects/save', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: USER_ID,
      title: taskText.value.slice(0, 40) || 'Untitled',
      task: taskText.value,
      code: codeText.textContent
    })
  });
  toast('✅ Проект сохранен');
  await loadProjects();
};

document.getElementById('btnDelete').onclick = async () => {
  if (!USER_ID || !currentProject) return;
  await fetch(API + '/projects/delete', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: USER_ID, project_id: currentProject })
  });
  currentProject = null;
  taskText.value = '';
  codeText.textContent = '';
  await loadProjects();
};

document.getElementById('btnSend').onclick = async () => {
  if (!USER_ID) return toast('Откройте Mini App из Telegram бота.');
  await fetch(API + '/projects/send_to_chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: USER_ID,
      title: taskText.value.slice(0, 40) || 'project',
      code: codeText.textContent
    })
  });
  toast('📤 Отправлено в чат');
};

document.getElementById('btnDonate').onclick = async () => {
  if (!USER_ID) return toast('Донат доступен только из Telegram.');
  toast('⭐ Используйте команду /donate в боте.');
};

async function loadAdminStats() {
  if (!USER_ID) return;
  const r = await fetch(API + '/admin/stats/' + USER_ID);
  if (r.status !== 200) return;
  IS_ADMIN = true;
  roleBadge.textContent = 'Admin mode';
  adminCard.style.display = 'block';
  const stats = await r.json();
  await fetch(API + '/admin/track_view', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: USER_ID })
  });
  adminStats.textContent = JSON.stringify(stats, null, 2);
}

loadProjects();
loadAdminStats();
</script>
</body>
</html>
"""


# ======================= TELEGRAM BOT =================
@dp.message(Command("start"))
async def start(msg: types.Message):
    await msg.answer(
        "💻 AI Code Studio Pro\n\n"
        "Открой Mini App кнопкой 🚀, генерируй и сохраняй проекты.\n"
        "Для поддержки проекта: /donate",
    )


@dp.message(Command("admin"))
async def admin_command(msg: types.Message):
    if not msg.from_user or not is_admin(msg.from_user.id):
        await msg.answer("⛔ Команда только для админов.")
        return

    stats = await get_admin_stats()
    events = "\n".join(
        [f"• {item['event']}: {item['count']}" for item in stats["events"][:8]]
    ) or "• нет событий"

    await msg.answer(
        "🛡 Admin статистика\n"
        f"Пользователей: {stats['users']}\n"
        f"Проектов: {stats['projects']}\n"
        f"Uptime: {stats['uptime_seconds']} сек\n\n"
        f"События:\n{events}"
    )


@dp.message(Command("donate"))
async def donate_command(msg: types.Message):
    await bot.send_invoice(
        chat_id=msg.chat.id,
        title="Support AI Code Studio",
        description="Поддержка развития бота ⭐",
        payload="donate_stars",
        currency="XTR",
        prices=[LabeledPrice(label="Support", amount=STAR_DONATE_AMOUNT)],
        provider_token="",
    )


@dp.pre_checkout_query()
async def process_pre_checkout_query(query: PreCheckoutQuery):
    await bot.answer_pre_checkout_query(query.id, ok=True)


@dp.message(lambda m: m.successful_payment is not None)
async def successful_payment_handler(msg: types.Message):
    await msg.answer("Спасибо за поддержку проекта! ⭐")


# ======================= WEBHOOK ======================
@app.post("/webhook")
async def telegram_webhook(request: Request):
    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    data = await request.json()
    update = Update.model_validate(data)
    asyncio.create_task(dp.feed_update(bot, update))
    return {"ok": True}


# ======================= STARTUP ======================
@app.on_event("startup")
async def on_startup():
    await init_db()

    if APP_URL and WEBHOOK_SECRET:
        await bot.set_webhook(
            url=f"{APP_URL}/webhook",
            secret_token=WEBHOOK_SECRET,
            drop_pending_updates=True,
        )

    if MINIAPP_URL:
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(
                text="🚀 Запустить",
                web_app=WebAppInfo(url=MINIAPP_URL),
            )
        )

    print("✅ Bot startup complete")
