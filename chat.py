"""
Sohbet modülü: kullanıcının doğal dil sorusunu DB içerikleriyle birlikte
Gemini'ye gönderir, Türkçe yanıt üretir.
"""
import logging

from google import genai
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

import db
from config import CHAT_CONTEXT_LIMIT, GEMINI_API_KEY, GEMINI_MODEL_FALLBACK, GEMINI_MODEL_PRIMARY

log = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (genai_errors.ServerError, genai_errors.ClientError)):
        return getattr(exc, "status_code", 0) in (429, 503)
    return False


def _build_context() -> str:
    """DB'deki tüm içerikleri Gemini'ye verilecek bağlam metnine dönüştür."""
    rows = db.get_all_contents(limit=CHAT_CONTEXT_LIMIT)
    if not rows:
        return "Veritabanında henüz kayıtlı içerik yok."

    parts = []
    for row in rows:
        tag_line = f"Etiketler: {row['tags']}\n" if row.get("tags") else ""
        note_line = f"Not: {row['note']}\n" if row.get("note") else ""
        parts.append(
            f"[ID:{row['id']}] {row['created_at'][:16]} | {row['platform']} | "
            f"Öncelik:{row['priority']}/10\n"
            f"Başlık: {row['title']}\n"
            f"URL: {row['url']}\n"
            f"{tag_line}{note_line}"
            f"Analiz: {row['analysis']}\n"
            f"Transkript özeti: {(row['transcript'] or '')[:300]}"
        )
    return "\n\n---\n\n".join(parts)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_gemini(model: str, prompt: str) -> str:
    response = gemini_client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()


def answer(user_question: str) -> str:
    """
    Kullanıcının sorusunu DB bağlamıyla birlikte Gemini'ye gönder,
    Türkçe yanıt döndür.
    """
    context = _build_context()
    log.info("Sohbet sorusu: %s", user_question)

    prompt = f"""Sen bir içerik küratör asistanısın. Kullanıcının kaydettiği video içeriklerini analiz eden bir botun parçasısın.

Aşağıda veritabanındaki kayıtlı içerikler var:

{context}

---

Kullanıcının sorusu: {user_question}

Lütfen soruyu sadece yukarıdaki veritabanı kayıtlarına dayanarak yanıtla.
- Türkçe yaz
- Kısa ve net ol
- Eğer soru belirli bir içerik veya platform hakkındaysa, ilgili kayıtları filtrele
- Eğer veri yoksa bunu açıkça söyle
- Markdown formatı kullanabilirsin (Telegram destekler)"""

    try:
        text = _call_gemini(GEMINI_MODEL_PRIMARY, prompt)
        log.info("Model kullanıldı: %s (%d karakter)", GEMINI_MODEL_PRIMARY, len(text))
    except genai_errors.ClientError as exc:
        if getattr(exc, "status_code", 0) in (404, 429):
            log.warning("Kota bitti (%s), fallback: %s", GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK)
            text = _call_gemini(GEMINI_MODEL_FALLBACK, prompt)
            log.info("Model kullanıldı: %s (%d karakter)", GEMINI_MODEL_FALLBACK, len(text))
        else:
            raise
    return text
