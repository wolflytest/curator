"""
SQLite veritabanı işlemleri.
"""
import sqlite3
import logging
from collections import Counter
from datetime import date
from pathlib import Path
from config import DB_PATH

log = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Tabloları oluştur, eksik sütunları ekle (migration)."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT NOT NULL,
                platform    TEXT NOT NULL,
                title       TEXT,
                transcript  TEXT,
                analysis    TEXT,
                priority    INTEGER,
                tags        TEXT,
                note        TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)
        # Eski veritabanlarına sütun ekle (varsa hata verme)
        for col, typedef in [("tags", "TEXT"), ("note", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE contents ADD COLUMN {col} {typedef}")
                log.info("Migration: '%s' sütunu eklendi.", col)
            except sqlite3.OperationalError:
                pass  # Sütun zaten var
        conn.commit()
    log.info("Veritabanı hazır: %s", DB_PATH)


def save_content(
    url: str,
    platform: str,
    title: str,
    transcript: str,
    analysis: str,
    priority: int,
    tags: list[str] | None = None,
    note: str | None = None,
) -> int:
    """Analiz edilmiş içeriği kaydet, yeni satırın id'sini döndür."""
    tags_str = ",".join(t.lower() for t in tags) if tags else ""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO contents
                (url, platform, title, transcript, analysis, priority, tags, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (url, platform, title, transcript, analysis, priority, tags_str, note or ""),
        )
        conn.commit()
        log.info("Kaydedildi → id=%s platform=%s tags=%s", cur.lastrowid, platform, tags_str)
        return cur.lastrowid


def get_all_contents(limit: int = 200) -> list[dict]:
    """Tüm kayıtları önceliğe göre sıralı getir (sohbet bağlamı için)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM contents
            ORDER BY priority DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_daily_contents(for_date: date | None = None) -> list[dict]:
    """Belirtilen güne ait (varsayılan: bugün) tüm içerikleri getir."""
    target = for_date or date.today()
    date_str = target.strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM contents
            WHERE date(created_at) = ?
            ORDER BY priority DESC, created_at ASC
            """,
            (date_str,),
        ).fetchall()
    return [dict(row) for row in rows]


def search_by_tag(tag: str) -> list[dict]:
    """Belirli bir tag'e sahip kayıtları önceliğe göre sıralı getir."""
    tag = tag.lower().lstrip("#")
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM contents
            WHERE ',' || tags || ',' LIKE ?
            ORDER BY priority DESC, created_at DESC
            """,
            (f"%,{tag},%",),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_content(content_id: int) -> bool:
    """Kaydı sil. Silinen satır varsa True döndür."""
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM contents WHERE id = ?", (content_id,))
        conn.commit()
    deleted = cur.rowcount > 0
    if deleted:
        log.info("Kayıt silindi: id=%d", content_id)
    else:
        log.warning("Silinecek kayıt bulunamadı: id=%d", content_id)
    return deleted


def get_stats() -> dict:
    """İstatistik verisini döndür."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM contents").fetchone()[0]
        avg_priority = conn.execute("SELECT AVG(priority) FROM contents").fetchone()[0]
        platforms = conn.execute(
            "SELECT platform, COUNT(*) as cnt FROM contents GROUP BY platform ORDER BY cnt DESC"
        ).fetchall()
        tag_rows = conn.execute(
            "SELECT tags FROM contents WHERE tags IS NOT NULL AND tags != ''"
        ).fetchall()

    # Tag frekansı
    tag_counter: Counter = Counter()
    for row in tag_rows:
        for tag in row["tags"].split(","):
            tag = tag.strip()
            if tag:
                tag_counter[tag] += 1

    return {
        "total": total,
        "avg_priority": round(avg_priority, 1) if avg_priority else 0,
        "platforms": [(r["platform"], r["cnt"]) for r in platforms],
        "top_tags": tag_counter.most_common(10),
    }
