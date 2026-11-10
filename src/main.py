
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters
)

from config import TELEGRAM_BOT_TOKEN
from bot.handlers import start_command, handle_message


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing!")

    app = Application.builder().token(
        TELEGRAM_BOT_TOKEN
    ).build()

    app.add_handler(
        CommandHandler("start", start_command)
    )

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("ShopAssist AI Bot is running...")

    app.run_polling()


if __name__ == "__main__":
    main()
