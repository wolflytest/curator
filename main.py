"""
Giriş noktası — botu başlatır.
Çalıştır: python main.py
"""
import logging
import sys

import db
from bot import build_application

# Loglama yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

log = logging.getLogger(__name__)


def main() -> None:
    log.info("Curator botu başlatılıyor...")
    db.init_db()
    app = build_application()
    log.info("Polling başladı. Çıkmak için Ctrl+C.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
