"""
Ana işlem hattı:
  URL → yt-dlp ile indir → ffmpeg ile ses ayır →
  PySceneDetect ile frame seç → Groq Whisper transkripsiyon →
  Gemini görsel + metin analizi → sonuç döndür
"""
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import yt_dlp
from groq import Groq
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from google import genai
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai import types as genai_types

from config import GEMINI_API_KEY, GROQ_API_KEY, MAX_FRAMES, TMP_DIR

log = logging.getLogger(__name__)

# API istemcilerini başlat
groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# Önce güçlü model denenir, kota bitince fallback devreye girer
GEMINI_MODEL_PRIMARY  = "gemini-3.1-flash-lite-preview"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"


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
    return "Diğer"


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
    }
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
    retry=retry_if_exception_type(Exception) if False else __import__("tenacity").retry_if_exception(_is_retryable),
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
        if getattr(exc, "status_code", 0) == 429:
            log.warning("Kota bitti (%s), fallback model deneniyor: %s",
                        GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK)
            analysis = _call_gemini(GEMINI_MODEL_FALLBACK, parts)
            log.info("Model kullanıldı: %s", GEMINI_MODEL_FALLBACK)
        else:
            raise

    # Öncelik skorunu metinden çek (örn. "7/10")
    match = re.search(r"ÖNCELİK SKORU.*?(\d+)/10", analysis, re.DOTALL | re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)/10", analysis)
    priority = int(match.group(1)) if match else 5
    priority = max(1, min(10, priority))

    log.info("Gemini analizi tamamlandı (öncelik=%d, %d karakter)", priority, len(analysis))
    return analysis, priority


def run(url: str, note: str = "") -> PipelineResult:
    """
    Tam işlem hattını çalıştır.
    Geçici dosyalar /tmp/curator/<uid>/ altında oluşturulur ve temizlenir.
    """
    platform = detect_platform(url)
    work_dir = Path(tempfile.mkdtemp(dir=TMP_DIR))
    log.info("Çalışma dizini: %s | Platform: %s", work_dir, platform)

    try:
        video_path, title = download_video(url, work_dir)
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
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.info("Geçici dosyalar temizlendi: %s", work_dir)
