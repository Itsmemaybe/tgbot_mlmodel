from flask import Flask, request
from telebot import TeleBot, types
import os, logging, sqlite3, bcrypt, requests
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---------- CONFIG ----------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("app")

TOKEN = os.environ.get("TELEGRAM_TOKEN", "8240498759:AAFDbNSZYNVxyceEKMQnewgWyCsYQVKAJbc")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "https://ivibl-46-252-243-58.a.free.pinggy.link/webhook")
MODEL_PATH = "people_tiger_classifier.h5"
DB_PATH = "users.db"

app = Flask(__name__)
bot = TeleBot(TOKEN, threaded=False)

# ---------- MODEL ----------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

# ---------- DATABASE ----------
def conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with conn() as c:
        cur = c.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                password_hash TEXT NOT NULL,
                prediction_count INTEGER DEFAULT 0,
                registered INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id INTEGER PRIMARY KEY,   -- уникальный идентификатор
                logged_in INTEGER DEFAULT 0,
                last_login TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS states (
                user_id INTEGER PRIMARY KEY,
                state TEXT
            )
        """)
        c.commit()

init_db()

# ---------- HELPERS ----------
def hash_password(p): return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode('utf-8')
def check_password(h, p): return bcrypt.checkpw(p.encode(), h.encode('utf-8'))
def is_password_strong(p): return len(p) >= 8 and any(ch.isdigit() for ch in p) and any(ch.isalpha() for ch in p)

def is_registered(chat_id: int) -> bool:
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT registered FROM users WHERE id=?", (chat_id,))
        r = cur.fetchone()
        return bool(r and r[0])

def current_user_id():
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT user_id FROM sessions WHERE logged_in=1 ORDER BY last_login DESC LIMIT 1")
        row = cur.fetchone()
        return row[0] if row else None

def set_current_user(user_id: int | None):
    """Залогинить указанного пользователя как текущего; если None — разлогинить всех."""
    with conn() as c:
        cur = c.cursor()
        # Разлогиниваем всех
        cur.execute("UPDATE sessions SET logged_in=0")
        if user_id is not None:
            sql = """
                INSERT INTO sessions (user_id, logged_in, last_login)
                VALUES (?, 1, ?)
                ON CONFLICT(user_id)
                DO UPDATE SET logged_in=1, last_login=excluded.last_login
            """
            cur.execute(sql, (user_id, datetime.utcnow().isoformat()))
        c.commit()


def set_state(chat_id: int, state: str):
    with conn() as c:
        cur = c.cursor()
        cur.execute("""
            INSERT INTO states (user_id, state)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET state=excluded.state
        """, (chat_id, state))
        c.commit()

def get_state(chat_id: int):
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT state FROM states WHERE user_id=?", (chat_id,))
        r = cur.fetchone()
        return r[0] if r else None

def clear_state(chat_id: int):
    with conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM states WHERE user_id=?", (chat_id,))
        c.commit()

def get_main_kb():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("Классифицировать изображение")
    kb.row("Моя статистика", "Помощь")
    kb.row("Выйти")
    return kb

REGISTER = "REGISTER"
LOGIN = "LOGIN"

# ---------- WEBHOOK ----------
@app.route("/webhook", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        update = types.Update.de_json(request.data.decode("utf-8"))
        bot.process_new_updates([update])
        return "", 200
    return "Unsupported", 415

# ---------- COMMANDS ----------
@bot.message_handler(commands=["start"])
def start_cmd(message):
    uid = current_user_id()
    if uid:
        bot.send_message(message.chat.id, "Привет снова 👋", reply_markup=get_main_kb())
    else:
        bot.send_message(
            message.chat.id,
            "Добро пожаловать! На устройстве никто не вошёл. Используйте /login или /register.",
            reply_markup=get_main_kb()
        )

@bot.message_handler(commands=["register"])
def register_cmd(message):
    if is_registered(message.chat.id):
        bot.send_message(message.chat.id, "Вы уже зарегистрированы! Используйте /login для входа.")
        return
    set_state(message.chat.id, REGISTER)
    bot.send_message(message.chat.id, "Введите пароль для регистрации (не менее 8 символов, включая буквы и цифры):")

@bot.message_handler(commands=["login"])
def login_cmd(message):
    set_state(message.chat.id, LOGIN)
    bot.send_message(message.chat.id, "Введите пароль для входа:")

@bot.message_handler(commands=["logout"])
def logout_cmd(message):
    set_current_user(None)
    clear_state(message.chat.id)
    bot.send_message(
        message.chat.id,
        "🚪 Вы вышли. Теперь классификация недоступна. Используйте /login, чтобы войти снова.",
        reply_markup=get_main_kb()
    )

# ---------- STATE HANDLERS ----------
@bot.message_handler(func=lambda m: get_state(m.chat.id) == REGISTER)
def handle_register(message):
    pwd = message.text.strip()
    if not is_password_strong(pwd):
        bot.send_message(message.chat.id, "Пароль слишком простой, попробуйте снова.")
        return
    # проверяем, нет ли пользователя с таким id
    if is_registered(message.chat.id):
        bot.send_message(message.chat.id, "Вы уже зарегистрированы! Используйте /login.")
        clear_state(message.chat.id)
        return

    with conn() as c:
        cur = c.cursor()
        cur.execute("""
            INSERT INTO users (id, username, password_hash, registered, created_at)
            VALUES (?, ?, ?, 1, ?)
        """, (message.chat.id, message.from_user.username, hash_password(pwd), datetime.utcnow().isoformat()))
        c.commit()
    clear_state(message.chat.id)
    bot.send_message(
        message.chat.id,
        "✅ Регистрация прошла успешно! Теперь используйте /login для входа.",
        reply_markup=get_main_kb()
    )

@bot.message_handler(func=lambda m: get_state(m.chat.id) == LOGIN)
def handle_login(message):
    pwd = message.text.strip()
    # находим пользователя с этим паролем
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT id, password_hash FROM users")
        rows = cur.fetchall()
    matched = None
    for uid, h in rows:
        if check_password(h, pwd):
            matched = uid
            break
    if matched:
        set_current_user(matched)
        clear_state(message.chat.id)
        bot.send_message(
            message.chat.id,
            "✅ Вы успешно вошли! Теперь можете отправить изображение для распознавания.",
            reply_markup=get_main_kb()
        )
    else:
        bot.send_message(message.chat.id, "❌ Неверный пароль. Попробуйте ещё раз или зарегистрируйтесь через /register.")

# ---------- BUTTONS ----------
@bot.message_handler(func=lambda m: m.text == "Выйти")
def btn_logout(message):
    set_current_user(None)
    bot.send_message(message.chat.id, "🚪 Вы вышли. Используйте /login, чтобы войти снова.",
                     reply_markup=get_main_kb())

@bot.message_handler(func=lambda m: m.text == "Классифицировать изображение")
def ask_image(message):
    if current_user_id() is None:
        bot.send_message(message.chat.id, "🔐 Сначала войдите с помощью /login.")
        return
    bot.send_message(message.chat.id, "Отправьте фото для классификации.")

@bot.message_handler(func=lambda m: m.text == "Моя статистика")
def stats_cmd(message):
    uid = current_user_id()
    if uid is None:
        bot.send_message(message.chat.id, "🔐 Сначала войдите с помощью /login.")
        return
    with conn() as c:
        cur = c.cursor()
        cur.execute("SELECT prediction_count FROM users WHERE id=?", (uid,))
        r = cur.fetchone()
        count = r[0] if r else 0
    bot.send_message(message.chat.id, f"📊 Для текущего пользователя выполнено {count} классификаций.")

@bot.message_handler(func=lambda m: m.text == "Помощь")
def help_cmd(message):
    bot.send_message(message.chat.id,
        "Команды:\n"
        "/start — главное меню\n"
        "/register — регистрация (без автоматического входа)\n"
        "/login — вход по паролю (делает пользователя текущим)\n"
        "/logout — выйти (сбрасывает текущего пользователя)\n"
        "Кнопка «Выйти» делает то же самое."
    )

# ---------- PHOTO ----------
@bot.message_handler(content_types=['photo'])
def classify_photo(message):
    uid = current_user_id()
    if uid is None:
        bot.send_message(message.chat.id, "🔐 Сначала войдите с помощью /login.")
        return
    file_info = bot.get_file(message.photo[-1].file_id)
    data = bot.download_file(file_info.file_path)
    tmp = f"temp_{message.chat.id}.jpg"
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        img = image.load_img(tmp, target_size=(200, 200))
        x = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
        pred = float(model.predict(x)[0][0])
        res = "🐯 Тигр" if pred > 0.5 else "🧍‍♂️ Человек"
        with conn() as c:
            cur = c.cursor()
            cur.execute("UPDATE users SET prediction_count = prediction_count + 1 WHERE id=?", (uid,))
            c.commit()
        bot.send_message(message.chat.id, f"Результат: {res}")
    except Exception as e:
        logger.exception("Ошибка распознавания: %s", e)
        bot.send_message(message.chat.id, "Ошибка обработки изображения.")
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

# ---------- RUN ----------
def set_webhook():
    try:
        requests.get(f"https://api.telegram.org/bot{TOKEN}/deleteWebhook", timeout=10)
        requests.get(f"https://api.telegram.org/bot{TOKEN}/setWebhook", params={"url": WEBHOOK_URL}, timeout=10)
    except Exception as e:
        logger.error("Webhook error: %s", e)

if __name__ == "__main__":
    set_webhook()
    app.run(host="0.0.0.0", port=8000)
