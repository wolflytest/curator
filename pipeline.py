"""
Ana işlem hattı:
  URL → yt-dlp ile indir → ffmpeg ile ses ayır →
  PySceneDetect ile frame seç → Groq Whisper transkripsiyon →
  Gemini görsel + metin analizi → sonuç döndür

Twitter/X desteği:
  Video tweet → normal pipeline
  Metin tweet → oEmbed API ile tweet metni çek → doğrudan Gemini'ye gönder
"""
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

import cv2
import yt_dlp
from groq import Groq
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from google import genai
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from google.genai import types as genai_types

from config import (
    COOKIE_FILE, GEMINI_API_KEY, GEMINI_MODEL_FALLBACK, GEMINI_MODEL_PRIMARY,
    GROQ_API_KEY, MAX_FRAMES, OPENCLAW_DIR, TMP_DIR,
)

log = logging.getLogger(__name__)

# API istemcilerini başlat
groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class PipelineResult:
    title: str
    platform: str
    transcript: str
    analysis: str
    priority: int  # 1-10


def detect_platform(url: str) -> str:
    """URL'den platform adını tahmin et."""
    url_lower = url.lower()
    if "instagram.com" in url_lower:
        return "Instagram"
    if "tiktok.com" in url_lower:
        return "TikTok"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "YouTube"
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "Twitter"
    return "Diğer"


def _expand_tco_urls(text: str, entities: dict) -> str:
    """t.co kısa linklerini entities'den alınan expanded URL ile değiştir."""
    for entry in entities.get("urls", []):
        short = entry.get("url", "")
        expanded = entry.get("expanded_url", "") or entry.get("display_url", "")
        if short and expanded:
            text = text.replace(short, expanded)
    return text


def fetch_tweet_text(url: str) -> tuple[str, str]:
    """
    Tweet metnini çek. Sırasıyla dört yöntem denenir:
      1. vxtwitter API (JSON) — makale önizlemesi dahil
      2. Twitter Syndication API — auth gerektirmez
      3. yt-dlp metadata (skip_download=True)
      4. Nitter HTML parse
    (tweet_metni, başlık) döndürür.
    """
    m = re.search(r"(?:twitter\.com|x\.com)/([^/?]+)/status/(\d+)", url)
    if not m:
        raise ValueError(f"Tweet URL'inden kullanıcı adı/ID çıkarılamadı: {url}")
    username, tweet_id = m.group(1), m.group(2)

    # --- Yöntem 1: vxtwitter API ---
    try:
        resp = httpx.get(
            f"https://api.vxtwitter.com/{username}/status/{tweet_id}",
            timeout=15,
            headers={"User-Agent": "curl/7.68.0"},
        )
        if "application/json" in resp.headers.get("content-type", ""):
            data = resp.json()
            raw_text = _expand_tco_urls(data.get("text", ""), data.get("entities", {}))
            article_preview = data.get("article", {}).get("preview_text", "")
            if article_preview:
                raw_text = (raw_text + "\n\n" + article_preview).strip()
            author_name = data.get("user_name", "") or username
            if raw_text.strip():
                log.info("vxtwitter ile tweet metni alındı (%d karakter)", len(raw_text))
                return raw_text.strip(), f"@{author_name} tweet'i"
        log.info("vxtwitter başarısız veya boş, syndication deneniyor...")
    except Exception as exc:
        log.info("vxtwitter başarısız (%s), syndication deneniyor...", exc)

    # --- Yöntem 2: Twitter Syndication API ---
    try:
        resp = httpx.get(
            f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=en&token=x",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = _expand_tco_urls(data.get("text", ""), data.get("entities", {}))
        author_name = data.get("user", {}).get("screen_name", "") or username
        if raw_text.strip():
            log.info("Syndication API ile tweet metni alındı (%d karakter)", len(raw_text))
            return raw_text.strip(), f"@{author_name} tweet'i"
        log.info("Syndication API boş yanıt, yt-dlp deneniyor...")
    except Exception as exc:
        log.info("Syndication API başarısız (%s), yt-dlp deneniyor...", exc)

    # --- Yöntem 3: yt-dlp metadata ---
    try:
        ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        tweet_text = info.get("description", "") or info.get("title", "")
        author = info.get("uploader", "") or info.get("channel", "")
        if tweet_text and tweet_text != author:
            title = f"@{author} tweet'i" if author else f"@{username} tweet'i"
            log.info("yt-dlp ile tweet metni alındı (%d karakter)", len(tweet_text))
            return tweet_text, title
        log.info("yt-dlp metadata boş, nitter deneniyor...")
    except Exception as exc:
        log.info("yt-dlp metadata başarısız (%s), nitter deneniyor...", exc)

    # --- Yöntem 4: Nitter ---
    nitter_url = f"https://nitter.poast.org/{username}/status/{tweet_id}"
    try:
        resp = httpx.get(nitter_url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "curator-bot/1.0"})
        resp.raise_for_status()

        class _NitterParser(HTMLParser):
            """Nitter'da tweet metni <div class="tweet-content ..."> içinde.
            Nested div'leri depth counter ile takip eder."""
            def __init__(self):
                super().__init__()
                self.texts: list[str] = []
                self._depth = 0  # tweet-content div'inden itibaren açık div sayısı

            def handle_starttag(self, tag, attrs):
                if self._depth > 0 and tag == "div":
                    self._depth += 1
                elif tag == "div" and "tweet-content" in dict(attrs).get("class", ""):
                    self._depth = 1

            def handle_endtag(self, tag):
                if self._depth > 0 and tag == "div":
                    self._depth -= 1

            def handle_data(self, data):
                if self._depth > 0 and data.strip():
                    self.texts.append(data.strip())

        parser = _NitterParser()
        parser.feed(resp.text)
        tweet_text = " ".join(parser.texts)
        if tweet_text:
            log.info("Nitter ile tweet metni alındı (%d karakter)", len(tweet_text))
            return tweet_text, f"@{username} tweet'i"
    except Exception as exc:
        log.warning("Nitter başarısız: %s", exc)

    raise ValueError(f"Tweet metni hiçbir yöntemle alınamadı: {url}")


def analyse_tweet_text(
    tweet_text: str,
    title: str,
    note: str = "",
) -> tuple[str, int]:
    """
    Metin içerikli tweet'i doğrudan Gemini'ye gönder.
    (analiz_metni, öncelik_skoru) döndürür.
    """
    log.info("Tweet metin analizi başlıyor...")

    prompt = f"""Sen bir içerik analiz uzmanısın. Aşağıdaki tweet'i Türkçe olarak analiz et:

## 1. GENEL BİLGİ
- Tweet konusu nedir (1-2 cümle)
- Hedef kitle kim
- Paylaşım amacı (bilgi vermek / görüş bildirmek / duyurmak / tartışmak)

## 2. TEMEL İÇERİK
- Tweet'te öne çıkan ana fikir veya argüman
- Belirtilen önemli veriler, rakamlar veya bağlantılar
- Varsa; bahsedilen araçlar, platformlar veya kaynaklar

## 3. İPUÇLARI VE ÖNEMLİ NOKTALAR
- Tweet'ten çıkarılabilecek pratik bilgiler veya öneriler
- Dikkat çekici veya takip edilmesi gereken noktalar

## 4. SONUÇ
Bu tweet'i okuyan kişi ne kazanır, ne öğrenir veya ne yapmalıdır

## 5. ÖNCELİK SKORU
Bu içeriğin öncelik skoru: X/10 (sadece rakam, örnek: 7/10)
Neden bu skoru verdin:

---
Tweet metni: {tweet_text}
Kullanıcı notu: {note or "(not yok)"}"""

    parts = [prompt]
    try:
        analysis = _call_gemini(GEMINI_MODEL_PRIMARY, parts)
        log.info("Model kullanıldı: %s", GEMINI_MODEL_PRIMARY)
    except genai_errors.ClientError as exc:
        if getattr(exc, "status_code", 0) in (404, 429):
            log.warning("Primary model başarısız (HTTP %s) → fallback: %s",
                        getattr(exc, "status_code", "?"), GEMINI_MODEL_FALLBACK)
            analysis = _call_gemini(GEMINI_MODEL_FALLBACK, parts)
            log.info("Model kullanıldı: %s", GEMINI_MODEL_FALLBACK)
        else:
            raise

    match = re.search(r"ÖNCELİK SKORU.*?(\d+)/10", analysis, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)/10", analysis)
    if not match:
        log.warning("Öncelik skoru çıkarılamadı, varsayılan 5 kullanılıyor.")
    priority = int(match.group(1)) if match else 5
    priority = max(1, min(10, priority))

    log.info("Tweet analizi tamamlandı (öncelik=%d, %d karakter)", priority, len(analysis))
    return analysis, priority


def download_video(url: str, work_dir: Path) -> tuple[Path, str]:
    """
    yt-dlp ile videoyu indir.
    (video_path, başlık) döndürür.
    """
    log.info("Video indiriliyor: %s", url)
    output_template = str(work_dir / "video.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            )
        },
    }
    is_instagram = "instagram.com" in url.lower()
    if is_instagram:
        ydl_opts.update({
            "sleep_interval_requests": 2,   # istek arası bekleme (rate-limit önlemi)
            "sleep_interval": 3,            # indirme arası bekleme
            "extractor_args": {"instagram": {"player_url": ["https://www.instagram.com"]}},
        })
    if COOKIE_FILE.exists():
        ydl_opts["cookiefile"] = str(COOKIE_FILE)
        log.info("Cookie dosyası kullanılıyor: %s", COOKIE_FILE)
    elif is_instagram:
        log.warning("Instagram cookie dosyası bulunamadı: %s — rate-limit riski yüksek", COOKIE_FILE)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Başlıksız")

    # İndirilen dosyayı bul
    video_files = list(work_dir.glob("video.*"))
    if not video_files:
        raise FileNotFoundError("Video dosyası indirilemedi.")
    return video_files[0], title


def extract_audio(video_path: Path, work_dir: Path) -> Path:
    """ffmpeg ile videodan ses ayır (mp3)."""
    audio_path = work_dir / "audio.mp3"
    log.info("Ses ayıklanıyor: %s", video_path.name)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "64k",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    return audio_path


def select_frames(video_path: Path, work_dir: Path, max_frames: int = MAX_FRAMES) -> list[Path]:
    """
    PySceneDetect ile sahne değişim noktalarını bul,
    en anlamlı max_frames kadar frame'i PNG olarak kaydet.
    """
    log.info("Frame seçimi yapılıyor (maks %d)...", max_frames)
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Sahne tespiti
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video, show_progress=False)
    scenes = scene_manager.get_scene_list()
    log.info("Tespit edilen sahne sayısı: %d", len(scenes))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved: list[Path] = []

    if scenes:
        # Her sahneden orta kare al, max_frames ile sınırla
        step = max(1, len(scenes) // max_frames)
        selected_scenes = scenes[::step][:max_frames]
        for i, (start, end) in enumerate(selected_scenes):
            mid_frame = (start.get_frames() + end.get_frames()) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                p = frames_dir / f"frame_{i:03d}.png"
                cv2.imwrite(str(p), frame)
                saved.append(p)
    else:
        # Sahne bulunamazsa eşit aralıklarla al
        interval = max(1, total_frames // max_frames)
        for i in range(min(max_frames, total_frames // interval)):
            pos = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                p = frames_dir / f"frame_{i:03d}.png"
                cv2.imwrite(str(p), frame)
                saved.append(p)

    cap.release()
    log.info("Kaydedilen frame sayısı: %d", len(saved))
    return saved


def transcribe_audio(audio_path: Path) -> str:
    """Groq Whisper ile ses dosyasını metne çevir."""
    log.info("Transkripsiyon başlıyor...")
    # Dosya yoksa veya 0 byte ise boş döndür
    if not audio_path.exists() or audio_path.stat().st_size < 1000:
        log.warning("Ses dosyası çok küçük veya yok, transkripsiyon atlanıyor.")
        return ""
    with open(audio_path, "rb") as f:
        response = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="text",
            language="tr",  # Otomatik algılama için kaldırılabilir
        )
    transcript = response if isinstance(response, str) else response.text
    log.info("Transkripsiyon tamamlandı (%d karakter)", len(transcript))
    return transcript


def _is_retryable(exc: BaseException) -> bool:
    """503 (geçici yük) ve 429 (rate limit) hatalarında retry yap."""
    if isinstance(exc, (genai_errors.ServerError, genai_errors.ClientError)):
        return getattr(exc, "status_code", 0) in (429, 503)
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_gemini(model: str, parts: list) -> str:
    """Tek bir Gemini isteği gönder, retry ile."""
    response = gemini_client.models.generate_content(
        model=model,
        contents=parts,
        config=genai_types.GenerateContentConfig(max_output_tokens=8192),
    )
    return response.text.strip()


def analyse_with_gemini(
    frames: list[Path],
    transcript: str,
    title: str,
    platform: str,
    note: str = "",
) -> tuple[str, int]:
    """
    Gemini 2.5 Flash-Lite'a frame'leri + transkripti gönder,
    (analiz_metni, öncelik_skoru) döndür.
    """
    log.info("Gemini analizi başlıyor (%d frame)...", len(frames))

    prompt = f"""Sen bir içerik analiz uzmanısın. Bu video için aşağıdaki analizin tamamını Türkçe olarak yap:

## 1. GENEL BİLGİ
- Video konusu nedir (1-2 cümle)
- Hedef kitle kim
- Videonun amacı (öğretmek / tanıtmak / satmak / eğlendirmek)

## 2. TÜM ADIMLAR (Tutorial ise)
Eğer video bir tutorial, rehber veya eğitim içeriği ise:
- Anlatılan TÜM adımları eksiksiz listele
- Hiçbir adımı atlama, özetleme veya birleştirme
- Her adımı ayrı madde olarak yaz
- Eğer 15 adım varsa 15 adımın hepsini yaz
- Her adımda kullanılan komut, prompt veya kod varsa, o adımın hemen altına girintili olarak kod bloğu içinde yaz:
  ```
  komut veya kod buraya, kelimesi kelimesine
  ```

## 3. EKRANDA GÖRÜNEN ÖNEMLİ İÇERİKLER
Frame'leri dikkatle incele. Adımlarla doğrudan bağlantısı olmayan ancak önemli olan ekran içeriklerini yaz:
- YAZ: Önemli URL'ler, araç isimleri, platform adları
- YAZ: Fiyatlar, süreler, rakamlar, istatistikler
- YAZ: Konuşmacının kasıtlı gösterdiği ama adım olmayan yazılar
- Konuşmacı bir prompt, komut veya kod yazıyorsa veya ekranda gösteriyorsa, o içeriği kod bloğu içinde (``` işaretleri arasında) kelimesi kelimesine yaz. Kısaltma veya özetleme yapma, tam olarak yaz.
- YAZMA: Adımlarda zaten yer alan komut ve kodlar (tekrar etme)
- YAZMA: Arka planda rastgele görünen site içerikleri
- YAZMA: Navigasyon menüleri, butonlar, genel UI elementleri
- YAZMA: Konuyla alakasız bildirimler veya pop-up'lar

## 4. KULLANILAN ARAÇLAR
Videoda bahsedilen veya gösterilen tüm araçlar, uygulamalar, platformlar ve servisler

## 5. İPUÇLARI VE UYARILAR
Konuşmacının özellikle vurguladığı ipuçları, dikkat edilmesi gereken noktalar ve uyarılar

## 6. SONUÇ
Bu videoyu uygulayan kişi ne elde eder, sonuç ne olacak

## 7. ÖNCELİK SKORU
Bu içeriğin öncelik skoru: X/10 (sadece rakam, örnek: 7/10)
Neden bu skoru verdin:

---
Transkript: {transcript or "(transkript mevcut değil)"}
Video başlığı: {title}
Platform: {platform}
Kullanıcı notu: {note or "(not yok)"}"""

    # Frame'leri inline bytes olarak ekle
    parts: list = [prompt]
    for frame_path in frames:
        with open(frame_path, "rb") as f:
            parts.append(
                genai_types.Part.from_bytes(
                    data=f.read(),
                    mime_type="image/png",
                )
            )

    try:
        analysis = _call_gemini(GEMINI_MODEL_PRIMARY, parts)
        log.info("Model kullanıldı: %s", GEMINI_MODEL_PRIMARY)
    except genai_errors.ClientError as exc:
        if getattr(exc, "status_code", 0) in (404, 429):
            log.warning("Primary model başarısız (HTTP %s) → fallback: %s",
                        getattr(exc, "status_code", "?"), GEMINI_MODEL_FALLBACK)
            analysis = _call_gemini(GEMINI_MODEL_FALLBACK, parts)
            log.info("Model kullanıldı: %s", GEMINI_MODEL_FALLBACK)
        else:
            raise

    # Öncelik skorunu metinden çek (örn. "7/10")
    match = re.search(r"ÖNCELİK SKORU.*?(\d+)/10", analysis, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)/10", analysis)
    if not match:
        log.warning("Öncelik skoru çıkarılamadı, varsayılan 5 kullanılıyor.")
    priority = int(match.group(1)) if match else 5
    priority = max(1, min(10, priority))

    log.info("Gemini analizi tamamlandı (öncelik=%d, %d karakter)", priority, len(analysis))
    return analysis, priority


def _url_to_id(url: str) -> str:
    """URL'den kısa, dosya adına uygun bir ID üret."""
    # YouTube: ?v=VIDEO_ID
    yt = re.search(r"[?&]v=([A-Za-z0-9_\-]{6,20})", url)
    if yt:
        return yt.group(1)
    # Diğerleri: son path segmentini al
    clean = url.rstrip("/").split("?")[0]
    segment = clean.split("/")[-1] or clean.split("/")[-2]
    safe = re.sub(r"[^\w\-]", "", segment)
    return safe[:40] or "unknown"


def _read_openclaw_token() -> str | None:
    """~/.openclaw/config.yaml dosyasından gateway token'ını oku."""
    config_path = Path.home() / ".openclaw" / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        # Olası key isimleri
        for key in ("token", "gateway_token", "api_token", "secret"):
            if data.get(key):
                return str(data[key])
    except Exception as exc:
        log.warning("OpenClaw token okunamadı: %s", exc)
    return None


def _notify_openclaw_gateway(filename: str) -> None:
    """
    OpenClaw gateway'e yeni dosyayı ingest etmesi için mesaj gönder.
    Hata olursa sadece loglar, pipeline'ı kesmez.
    """
    gateway_url = "http://localhost:18789/api/agent/message"
    token = _read_openclaw_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body = {"message": f"Ingest new file: ~/curator/openclaw_knowledge/{filename} into knowledge base"}
    try:
        resp = httpx.post(gateway_url, json=body, headers=headers, timeout=10)
        resp.raise_for_status()
        log.info("OpenClaw gateway bildirildi: %s (HTTP %d)", filename, resp.status_code)
    except httpx.ConnectError:
        log.warning("OpenClaw gateway bağlantısı kurulamadı (localhost:18789 çalışıyor mu?)")
    except Exception as exc:
        log.warning("OpenClaw gateway bildirimi başarısız: %s", exc)


def save_to_openclaw(result: PipelineResult, url: str) -> Path:
    """
    Analiz sonucunu openclaw_knowledge/ dizinine Markdown olarak kaydet,
    ardından OpenClaw gateway'e ingest bildirimi gönder.
    Dosya adı: {platform}_{id}_{tarih}.md
    """
    from datetime import date
    today = date.today().strftime("%Y%m%d")
    content_id = _url_to_id(url)
    platform_slug = re.sub(r"[^\w]", "", result.platform).lower()
    filename = f"{platform_slug}_{content_id}_{today}.md"
    filepath = OPENCLAW_DIR / filename

    content = f"""# {result.title}

| Alan | Değer |
|------|-------|
| Platform | {result.platform} |
| URL | {url} |
| Tarih | {date.today().isoformat()} |
| Öncelik | {result.priority}/10 |

---

{result.analysis}
"""
    filepath.write_text(content, encoding="utf-8")
    log.info("OpenClaw'a kaydedildi: %s", filepath.name)
    _notify_openclaw_gateway(filename)
    return filepath


def prepare_audio_for_recognition(url: str) -> tuple[bytes, str]:
    """
    Şarkı tanıma için video indir ve ses ayıkla.
    (audio_bytes, başlık) döndürür; geçici dosyaları temizler.
    """
    work_dir = Path(tempfile.mkdtemp(dir=TMP_DIR))
    try:
        video_path, title = download_video(url, work_dir)
        audio_path = extract_audio(video_path, work_dir)
        return audio_path.read_bytes(), title
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info("Geçici dosyalar temizlendi: %s", work_dir)


def run(url: str, note: str = "") -> PipelineResult:
    """
    Tam işlem hattını çalıştır.
    Geçici dosyalar /tmp/curator/<uid>/ altında oluşturulur ve temizlenir.

    Twitter/X için:
      - Önce video indirme denenir.
      - Video yoksa (DownloadError) tweet metni oEmbed API ile çekilir.
    """
    platform = detect_platform(url)
    work_dir = Path(tempfile.mkdtemp(dir=TMP_DIR))
    log.info("Çalışma dizini: %s | Platform: %s", work_dir, platform)

    try:
        if platform == "Twitter":
            result = _run_twitter(url, note, platform, work_dir)
        else:
            video_path, title = download_video(url, work_dir)
            audio_path = extract_audio(video_path, work_dir)
            frames = select_frames(video_path, work_dir)
            transcript = transcribe_audio(audio_path)
            analysis, priority = analyse_with_gemini(frames, transcript, title, platform, note)
            result = PipelineResult(
                title=title,
                platform=platform,
                transcript=transcript,
                analysis=analysis,
                priority=priority,
            )
        save_to_openclaw(result, url)
        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info("Geçici dosyalar temizlendi: %s", work_dir)


def _run_twitter(url: str, note: str, platform: str, work_dir: Path) -> PipelineResult:
    """Twitter/X için: video varsa normal pipeline, yoksa metin analizi."""
    # Video tweet dene
    try:
        video_path, title = download_video(url, work_dir)
        log.info("Twitter video tweet tespit edildi, normal pipeline çalışıyor.")
        audio_path = extract_audio(video_path, work_dir)
        frames = select_frames(video_path, work_dir)
        transcript = transcribe_audio(audio_path)
        analysis, priority = analyse_with_gemini(frames, transcript, title, platform, note)
        return PipelineResult(
            title=title,
            platform=platform,
            transcript=transcript,
            analysis=analysis,
            priority=priority,
        )
    except yt_dlp.utils.DownloadError as exc:
        log.info("Twitter'da video indirilemedi (%s), metin olarak işleniyor.", exc)

    # Metin tweet: içerik çek
    tweet_text, title = fetch_tweet_text(url)
    if not tweet_text:
        raise ValueError("Tweet metni alınamadı ve video da bulunamadı.")

    analysis, priority = analyse_tweet_text(tweet_text, title, note)
    return PipelineResult(
        title=title,
        platform=platform,
        transcript=tweet_text,
        analysis=analysis,
        priority=priority,
    )
