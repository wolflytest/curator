"""
Telegram botu:
  - Gelen linkler → pipeline çalıştır → DB'ye kaydet
  - Her gece 21:00 → günlük özet gönder
  - /ozet, /soru, /istatistik komutları
"""
import asyncio
import logging
import re
import shutil
from functools import partial

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import chat
import db
import pipeline as pl
from config import (
    SUMMARY_HOUR,
    SUMMARY_MINUTE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_USER_ID,
)

log = logging.getLogger(__name__)

# Desteklenen platform URL'leri
# [^\s>)\]\}]* — boşluk ve kapanış karakterlerini almaz (parantez, köşeli parantez vs.)
URL_RE = re.compile(
    r"https?://(?:www\.)?"
    r"(?:instagram\.com|tiktok\.com|youtube\.com|youtu\.be|twitter\.com|x\.com)"
    r"[^\s>)\]\}]*",
    re.IGNORECASE,
)

# #tag tespiti (Unicode harf/rakam destekli)
TAG_RE = re.compile(r"#([\w\u00c0-\u024f\u0400-\u04ff]+)", re.UNICODE)


def _parse_tags_and_note(text: str, urls: list[str]) -> tuple[list[str], str]:
    """
    Mesaj metninden tag listesi ve not çıkar.
    - Tags: #kelime şeklindeki ifadeler (lowercase normalize edilir)
    - Not: URL'ler ve tag'ler çıkarıldıktan sonra kalan metin
    """
    tags = [t.lower() for t in TAG_RE.findall(text)]
    note = text
    for url in urls:
        note = note.replace(url, "")
    note = TAG_RE.sub("", note).strip()
    return tags, note


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    """
    Metni max_len karakterlik parçalara böl.
    Satır ortasında kesmez; her zaman satır sonundan böler.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        # Tek satır limitin üzerindeyse zorla böl
        if len(line) > max_len:
            if current:
                chunks.append("".join(current))
                current, current_len = [], 0
            for i in range(0, len(line), max_len):
                chunks.append(line[i:i + max_len])
            continue

        if current_len + len(line) > max_len:
            chunks.append("".join(current))
            current, current_len = [], 0

        current.append(line)
        current_len += len(line)

    if current:
        chunks.append("".join(current))

    return chunks


def _auth(update: Update) -> bool:
    return update.effective_user.id == TELEGRAM_USER_ID


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Kullanıcıdan gelen mesajı işle: link → pipeline, yoksa sohbet."""
    if not _auth(update):
        log.warning("Yetkisiz erişim denemesi: %s", update.effective_user.id)
        return

    text = update.message.text or ""
    urls = URL_RE.findall(text)

    if not urls:
        await handle_chat(update, text)
        return

    tags, note = _parse_tags_and_note(text, urls)
    log.info("Tags: %s | Not: %s", tags, note[:60] if note else "—")

    if "sarki" in tags:
        for url in urls:
            await _recognize_song(update, url)
        return

    for url in urls:
        status_msg = await update.message.reply_text(
            f"⏳ İşleniyor: `{url}`",
            parse_mode="Markdown",
        )
        try:
            log.info("Pipeline başlatılıyor: %s", url)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, pl.run, url, note)

            db.save_content(
                url=url,
                platform=result.platform,
                title=result.title,
                transcript=result.transcript,
                analysis=result.analysis,
                priority=result.priority,
                tags=tags,
                note=note,
            )

            tag_line = ("🏷 " + "  ".join(f"#{t}" for t in tags) + "\n") if tags else ""
            note_line = (f"📌 _{note}_\n") if note else ""

            full_text = (
                f"✅ Kaydedildi\n\n"
                f"*{result.title}*\n"
                f"📱 {result.platform} | 🏆 Öncelik: {result.priority}/10\n"
                f"{tag_line}{note_line}\n"
                f"{result.analysis}"
            )
            chunks = _split_message(full_text)
            await status_msg.edit_text(chunks[0], parse_mode="Markdown")
            for chunk in chunks[1:]:
                await update.message.reply_text(chunk, parse_mode="Markdown")
            log.info("Başarıyla kaydedildi: %s (%d parça)", result.title, len(chunks))

        except Exception as exc:
            log.error("Pipeline hatası: %s", exc, exc_info=True)
            await status_msg.edit_text(
                f"❌ Hata oluştu:\n`{exc}`",
                parse_mode="Markdown",
            )


async def _recognize_song(update: Update, url: str) -> None:
    """Video/reels'deki şarkıyı Shazam ile tanı ve kullanıcıya bildir."""
    from shazamio import Shazam

    status_msg = await update.message.reply_text(
        f"🎵 Şarkı aranıyor: `{url}`",
        parse_mode="Markdown",
    )
    try:
        loop = asyncio.get_event_loop()
        audio_bytes, title = await loop.run_in_executor(
            None, pl.prepare_audio_for_recognition, url
        )
        shazam = Shazam()
        out = await shazam.recognize(audio_bytes)
        track = out.get("track", {})
        if not track:
            await status_msg.edit_text("❓ Şarkı tanınamadı.")
            return

        song_title = track.get("title", "Bilinmiyor")
        artist = track.get("subtitle", "Bilinmiyor")
        # Şarkı sayfası linki (varsa)
        share = track.get("share", {})
        song_url = share.get("href", "")
        url_line = f"\n🔗 {song_url}" if song_url else ""

        await status_msg.edit_text(
            f"🎵 *{song_title}*\n🎤 {artist}{url_line}",
            parse_mode="Markdown",
        )
        log.info("Şarkı tanındı: %s — %s", song_title, artist)
    except Exception as exc:
        log.error("Şarkı tanıma hatası: %s", exc, exc_info=True)
        await status_msg.edit_text(f"❌ Hata: `{exc}`", parse_mode="Markdown")


async def handle_chat(update: Update, text: str) -> None:
    """Link içermeyen mesajı sohbet olarak işle."""
    log.info("Sohbet sorusu alındı: %s", text[:80])
    thinking_msg = await update.message.reply_text("💭 Düşünüyorum...")
    try:
        loop = asyncio.get_event_loop()
        reply = await loop.run_in_executor(None, chat.answer, text)
        await thinking_msg.edit_text(reply, parse_mode="Markdown")
    except Exception as exc:
        log.error("Sohbet hatası: %s", exc, exc_info=True)
        await thinking_msg.edit_text(f"❌ Hata: `{exc}`", parse_mode="Markdown")


async def cmd_ozet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/ozet — günlük özeti manuel tetikler."""
    if not _auth(update):
        return
    await send_daily_summary(context.application)


async def cmd_soru(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/soru <metin> — DB kayıtlarına bakarak Gemini ile Türkçe cevap verir."""
    if not _auth(update):
        return
    soru = " ".join(context.args) if context.args else ""
    if not soru:
        await update.message.reply_text("❓ Kullanım: `/soru bugün ne kaydettim?`", parse_mode="Markdown")
        return
    await handle_chat(update, soru)


async def cmd_istatistik(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/istatistik — toplam kayıt, platform dağılımı, ortalama öncelik, en çok kullanılan tag'ler."""
    if not _auth(update):
        return

    stats = db.get_stats()

    platform_lines = "\n".join(
        f"  • {platform}: {cnt} içerik"
        for platform, cnt in stats["platforms"]
    ) or "  — veri yok"

    tag_lines = "\n".join(
        f"  • #{tag}: {cnt}x"
        for tag, cnt in stats["top_tags"]
    ) or "  — henüz tag eklenmemiş"

    text = (
        f"📊 *Küratör İstatistikleri*\n\n"
        f"📦 Toplam içerik: *{stats['total']}*\n"
        f"⭐ Ortalama öncelik: *{stats['avg_priority']}/10*\n\n"
        f"📱 *Platform Dağılımı:*\n{platform_lines}\n\n"
        f"🏷 *En Çok Kullanılan Tag'ler:*\n{tag_lines}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_tag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/tag <etiket> — bu tag'e sahip kayıtları listeler."""
    if not _auth(update):
        return
    tag = (context.args[0] if context.args else "").lstrip("#").lower()
    if not tag:
        await update.message.reply_text("❓ Kullanım: `/tag python`", parse_mode="Markdown")
        return
    rows = db.search_by_tag(tag)
    if not rows:
        await update.message.reply_text(f"🔍 `#{tag}` ile eşleşen kayıt yok.", parse_mode="Markdown")
        return
    lines = [f"🏷 *#{tag}* — {len(rows)} kayıt\n"]
    for row in rows:
        lines.append(
            f"*[{row['id']}] {row['title']}*\n"
            f"📱 {row['platform']} | 🏆 {row['priority']}/10 | 🕐 {row['created_at'][:10]}\n"
            f"🔗 {row['url']}\n"
        )
    for chunk in _split_message("\n".join(lines)):
        await update.message.reply_text(chunk, parse_mode="Markdown")


async def cmd_sil(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/sil <id> — kaydı veritabanından siler."""
    if not _auth(update):
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("❓ Kullanım: `/sil 42`", parse_mode="Markdown")
        return
    content_id = int(context.args[0])
    if db.delete_content(content_id):
        await update.message.reply_text(f"🗑 Kayıt #{content_id} silindi.")
    else:
        await update.message.reply_text(f"❌ #{content_id} ID'li kayıt bulunamadı.")


async def send_daily_summary(app: Application) -> None:
    """Günlük özet mesajını hazırla ve gönder."""
    log.info("Günlük özet gönderiliyor...")
    rows = db.get_daily_contents()

    if not rows:
        await app.bot.send_message(chat_id=TELEGRAM_USER_ID, text="📭 Bugün kayıtlı içerik yok.")
        return

    lines = [f"📊 *Günlük İçerik Özeti*\n\nToplam {len(rows)} içerik\n{'─' * 28}"]

    for i, row in enumerate(rows, 1):
        tag_line = ("🏷 " + "  ".join(f"#{t}" for t in row["tags"].split(",") if t) + "\n") if row["tags"] else ""
        note_line = (f"📌 _{row['note']}_\n") if row["note"] else ""
        lines.append(
            f"\n*{i}. {row['title']}*\n"
            f"📱 {row['platform']} | 🕐 {row['created_at'][11:16]} | 🏆 {row['priority']}/10\n"
            f"{tag_line}{note_line}"
            f"🔗 {row['url']}\n\n"
            f"{row['analysis']}\n"
            f"{'─' * 28}"
        )

    full_text = "\n".join(lines)
    for chunk in _split_message(full_text):
        await app.bot.send_message(chat_id=TELEGRAM_USER_ID, text=chunk, parse_mode="Markdown")

    log.info("Günlük özet gönderildi (%d içerik)", len(rows))


def build_application() -> Application:
    """Telegram uygulamasını ve zamanlayıcıyı yapılandır."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("ozet", cmd_ozet))
    app.add_handler(CommandHandler("soru", cmd_soru))
    app.add_handler(CommandHandler("istatistik", cmd_istatistik))
    app.add_handler(CommandHandler("tag", cmd_tag))
    app.add_handler(CommandHandler("sil", cmd_sil))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        send_daily_summary,
        trigger="cron",
        hour=SUMMARY_HOUR,
        minute=SUMMARY_MINUTE,
        kwargs={"app": app},
        id="daily_summary",
        name="Günlük Özet",
        replace_existing=True,
    )

    async def on_startup(application: Application) -> None:
        scheduler.start()
        log.info("Zamanlayıcı başlatıldı — özet saati: %02d:%02d", SUMMARY_HOUR, SUMMARY_MINUTE)

    async def on_shutdown(application: Application) -> None:
        scheduler.shutdown(wait=False)
        log.info("Zamanlayıcı durduruldu.")

    app.post_init = on_startup
    app.post_shutdown = on_shutdown

    return app
