import re
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

class AnswerGenerator:

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing!")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate(self, question: str, retrieved_docs: list):
        if not retrieved_docs:
            if re.search(r"[\u0600-\u06FF]", question):
                return (
                    "عذراً، لم أجد معلومات كافية للإجابة على سؤالك. 🙏\n"
                    "يمكنك التواصل مع خدمة عملاء Nile Store للمساعدة."
                )
            return (
                "I'm sorry, but I couldn't find enough information to answer your question. 🙏\n"
                "Please reach out to Nile Store customer support for further assistance."
            )

        context_parts = []

        for index, doc in enumerate(retrieved_docs, start=1):
            context_parts.append(
                f"""
Document {index}:
Content: {doc['text']}
Source: {doc['source']}
Page: {doc['page']}
"""
            )

        context = "\n".join(context_parts)

        system_prompt = """
You are ShopAssist AI, the customer support assistant for Nile Store.
Your job is to answer customer questions using ONLY the information provided in the context.

CRITICAL LANGUAGE RULE (TOP PRIORITY):
- Detect the exact language of the Customer question.
- IF THE QUESTION IS IN ENGLISH -> YOU MUST ANSWER ENTIRELY IN ENGLISH.
- IF THE QUESTION IS IN ARABIC -> YOU MUST ANSWER IN NATURAL EGYPTIAN ARABIC.
- NEVER ANSWER AN ENGLISH QUESTION IN ARABIC OR VICE VERSA.

STRICT PERSONA & CONTENT RULES:
1. Never invent prices, policies, products, or delivery details.
2. If a specific service or feature (e.g. installment plans, specific payment method) is not mentioned in the context, treat it as currently unavailable at the store. In English: "Currently, this service is not available at our store." In Arabic: "حالياً هذه الخدمة غير متوفرة في المتجر."
3. NEVER refer to yourself in the third person as "the robot", "the chatbot", "the pilot", or "الروبوت". Speak naturally in the first person ("I can help...", "We accept...", "نحن نوفر...").
4. STRICT PERSONA RULE: NEVER mention words like "documents", "files", "PDFs", "source", "مستندات", "وثائق", "الملفات", or "السياق" in your response. Speak naturally and authentically as a professional customer support representative for Nile Store.
5. Never claim that an order was placed, a refund was approved, or a payment was processed.
6. Only provide contact details explicitly found in the retrieved context. If contact information is unavailable, advise the customer to check the official store website.
7. Be concise, clear, and helpful.
"""

        user_prompt = f"""
Context:
{context}

Customer question:
{question}

Instructions: Answer the customer question based only on the context above.
Answer strictly in the SAME language as the Customer question (if English -> reply in English; if Arabic -> reply in Arabic).
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=500
        )

        answer = completion.choices[0].message.content

        return answer.strip()

if __name__ == "__main__":
    from rag.retriever import Retriever

    retriever = Retriever()
    generator = AnswerGenerator()

    question = "كام سعر الماوس اللاسلكي؟"

    print("Question:", question)

    documents = retriever.search(question)

    answer = generator.generate(question, documents)

    print("\nShopAssist AI:")
    print(answer)
