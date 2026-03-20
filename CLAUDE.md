# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje

`curator` — Instagram/TikTok/YouTube linklerini Telegram üzerinden alıp indiren, analiz eden ve gece 21:00'de özet gönderen Python tabanlı içerik küratör botudur.

## Komutlar

```bash
# Botu çalıştır
source venv/bin/activate
python main.py

# Paketleri kur
venv/bin/pip install -r requirements.txt

# Tek modülü import test et
venv/bin/python -c "import pipeline"
```

## Ortam

- Python 3.12, venv: `venv/`
- `.env` dosyası proje kökünde: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_USER_ID`, `GROQ_API_KEY`, `GEMINI_API_KEY`
- Geçici dosyalar: `/tmp/curator/<uid>/` — işlem bitince otomatik silinir
- Veritabanı: `curator.db` (SQLite, `db.init_db()` ile otomatik oluşur/migrate edilir)

## Mimari

```
main.py       → db.init_db() + bot.build_application() + run_polling()
bot.py        → Telegram handler + APScheduler (21:00 özet) + /ozet /soru /istatistik komutları
pipeline.py   → URL → yt-dlp → ffmpeg → PySceneDetect → Groq Whisper → Gemini
chat.py       → DB bağlamı + kullanıcı sorusu → Gemini → Türkçe yanıt
db.py         → SQLite: save_content(), get_daily_contents(), get_all_contents(), get_stats()
config.py     → .env yükle, sabitler (MAX_FRAMES=8, TMP_DIR, özet saati)
```

### İşlem Hattı (`pipeline.run(url, note="")`)

1. `detect_platform()` — URL'den Instagram/TikTok/YouTube tespit et
2. `download_video()` — yt-dlp ile mp4 indir
3. `extract_audio()` — ffmpeg ile 16kHz mono mp3 ayır
4. `select_frames()` — PySceneDetect (ContentDetector) ile sahne sınırlarından max 8 frame seç; sahne bulunamazsa eşit aralıklı
5. `transcribe_audio()` — Groq `whisper-large-v3-turbo` ile transkripsiyon
6. `analyse_with_gemini()` — Gemini'ye frame'ler (inline bytes) + transkript + kullanıcı notu → JSON yanıt → `(analiz_str, öncelik_int)`

### Gemini Analiz Çıktısı (JSON alanları)

`ozet`, `adimlar` (tutorial ise TÜM adımlar eksiksiz), `araclar`, `rakamlar`, `ipuclari_ve_uyarilar`, `sonuc`, `hedef_kitle`, `eylem_gerektirir_mi`, `oncelik_skoru` (1-10), `oncelik_nedeni`

`max_output_tokens=4096` — tutorial videolarda adımların kesilmemesi için.

### Önemli Kararlar

- **Gemini SDK**: `google-generativeai` deprecated; `from google import genai` (google-genai paketi) kullanılıyor. Model: `gemini-2.5-flash-lite`
- **Retry**: `analyse_with_gemini()` ve `chat.answer()` üzerinde tenacity ile 503 hatalarına karşı exponential backoff (5s→10s→20s→hata)
- **Blocking pipeline**: `pipeline.run()` CPU-bound; `asyncio.get_event_loop().run_in_executor(None, pl.run, url, note)` ile çalıştırılıyor
- **Tag & Not parse**: `bot._parse_tags_and_note()` — URL ve `#tag`'ler çıkarıldıktan sonra kalan metin `note` olur; Gemini prompt'una eklenir
- **DB migration**: `init_db()` mevcut tabloya `ALTER TABLE` ile `tags`/`note` sütunları ekler, hata verme

## MCP Sunucuları

Bu proje için kayıtlı MCP sunucuları (`claude mcp list` ile doğrulanabilir):

| Sunucu | Komut |
|--------|-------|
| `filesystem` | `npx -y @modelcontextprotocol/server-filesystem /home/emre/curator` |
| `sqlite` | `/home/emre/curator/venv/bin/mcp-server-sqlite --db-path curator.db` |
| `sequentialthinking` | `npx -y @modelcontextprotocol/server-sequential-thinking` |
| `context7` | `npx -y @upstash/context7-mcp` |

`mcp-server-sqlite` npm'de mevcut değil; venv içine `pip install mcp-server-sqlite` ile kurulur.

### Telegram Komutları

| Komut | İşlev |
|-------|-------|
| Link gönder | Pipeline başlatır |
| Link + `#tag` | Tag'li kaydeder |
| Link + metin | Metni not olarak kaydeder ve Gemini'ye gönderir |
| Normal mesaj | `chat.answer()` ile DB'ye bakarak yanıtlar |
| `/ozet` | Günlük özeti hemen gönderir |
| `/soru <metin>` | DB'ye bakarak Gemini ile yanıtlar |
| `/istatistik` | Toplam kayıt, platform dağılımı, tag frekansı |
