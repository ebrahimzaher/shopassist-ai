
from telegram import Update
from telegram.ext import ContextTypes


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "أهلاً بيك في Nile Store! 👋\n"
        "أنا المساعد الذكي للمتجر.\n"
        "اسألني عن المنتجات أو الشحن أو الاسترجاع."
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user_message = update.message.text

    await update.message.reply_text(
        f"استلمت رسالتك: {user_message}\n\n"
        "لسه بتعلم أجاوب على أسئلة المتجر 😊"
    )
