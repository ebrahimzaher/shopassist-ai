import asyncio
import logging
import re
from telegram import Update
from telegram.ext import ContextTypes
from rag import Retriever
from rag import AnswerGenerator

logger = logging.getLogger(__name__)

retriever = None
generator = None

def initialize_services():
    global retriever, generator

    print("Initializing RAG services...")

    retriever = Retriever()
    generator = AnswerGenerator()

    print("RAG services initialized successfully!")

def handle_general_question(question: str):
    normalized = question.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    responses = {
        # Identity
        "who are you": (
            "I'm ShopAssist AI, your customer support "
            "assistant for Nile Store."
        ),
        "what is your name": (
            "I'm ShopAssist AI, your customer support "
            "assistant for Nile Store."
        ),
        "what can you do": (
            "I can help you with products, prices, shipping, "
            "returns, and payment methods."
        ),

        # Greetings
        "hello": "Hello! How can I help you today?",
        "hi": "Hi! How can I help you today?",
        "hey": "Hey! How can I help you today?",
        "good morning": "Good morning! How can I help you?",
        "good evening": "Good evening! How can I help you?",

        # Small talk
        "how are you": (
            "I'm doing great, thanks for asking! "
            "How can I help you today?"
        ),
        "how are you doing": (
            "I'm doing well! Thanks for asking. "
            "How can I help you?"
        ),
        "okay": "Great! Let me know if you need any help.",
        "ok": "Great! Let me know if you need any help.",
        "thanks": "You're welcome! Happy to help.",
        "thank you": "You're welcome! Happy to help.",
        "okay thanks": "You're welcome! Let me know if you need anything.",
        "ok thanks": "You're welcome! Let me know if you need anything.",
        "thank you very much": "You're very welcome!",

        # Arabic
        "انت مين": "أنا ShopAssist AI، مساعد خدمة العملاء لمتجر Nile Store.",
        "اسمك ايه": "أنا ShopAssist AI، مساعد خدمة العملاء لمتجر Nile Store.",
        "بتعمل ايه": "أقدر أساعدك في المنتجات والأسعار والشحن والاسترجاع وطرق الدفع.",
        "ازيك": "الحمد لله، تمام! أقدر أساعدك في إيه؟",
        "شكرا": "العفو! سعيد إني أقدر أساعدك.",
        "تمام": "تمام! لو محتاج أي مساعدة أنا موجود.",
    }

    if "thanks" in normalized or "thank you" in normalized:
        return "You're welcome! Let me know if you need anything."
    return responses.get(normalized)

def process_question(question: str) -> str:
    documents = retriever.search(question)

    answer = generator.generate(question, documents)

    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)

    return answer

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    welcome_message = """
أهلاً بيك في Nile Store! 👋

أنا ShopAssist AI، المساعد الذكي للمتجر.

تقدر تسألني عن:
📦 المنتجات والأسعار
🚚 الشحن والتوصيل
🔄 الاسترجاع والاسترداد
💳 طرق الدفع

اسألني أي سؤال!
"""
    await update.message.reply_text(welcome_message)

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    question = update.message.text.strip()

    if not question:
        return

    if retriever is None or generator is None:
        await update.message.reply_text(
            "المساعد غير جاهز حاليًا. حاول مرة أخرى بعد قليل."
        )
        return

    await update.message.chat.send_action("typing")

    try:
        general_answer = handle_general_question(question)

        if general_answer:
            answer = general_answer
        else:
            answer = await asyncio.to_thread(
                process_question,
                question
            )

        await update.message.reply_text(answer)

    except Exception:
        logger.exception("Error while processing user question")

        await update.message.reply_text(
            "حصلت مشكلة أثناء معالجة سؤالك. حاول مرة تانية بعد قليل."
        )
