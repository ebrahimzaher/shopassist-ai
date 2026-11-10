import logging
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters
)
from config import TELEGRAM_BOT_TOKEN

from bot.handlers import (
    start_command,
    handle_message,
    initialize_services
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing!")

    initialize_services()

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start_command)
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("ShopAssist AI Telegram Bot is running...")

    application.run_polling()

if __name__ == "__main__":
    main()
