# motiondetector/core/logging_setup.py
import logging
import os
import sys
import time
import shutil
import gzip
import pathlib
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler


def setup_logging(place: str,
                  cuda: str = None,
                  base_dir: str = None,
                  level: int = logging.INFO,
                  rotate: str = "time",
                  when: str = "midnight",
                  interval: int = 1,
                  backup: int = 14,
                  max_mb: int = 100,
                  compress: bool = True,
                  console: bool = True):
    """
    logs/<place>/disinfection_service.log 按天或按大小轮转，并压缩历史文件
    """
    # 默认放到 “项目根目录/logs”
    if base_dir is None:
        project_root = pathlib.Path(__file__).resolve().parents[2]
        base_dir = project_root / "logs"

    log_dir = pathlib.Path(base_dir) / place
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "disinfection_service.log"

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    if rotate == "time":
        fh = TimedRotatingFileHandler(
            log_file, when=when, interval=interval,
            backupCount=backup, encoding="utf-8", delay=True, utc=False
        )
    else:
        fh = RotatingFileHandler(
            log_file, maxBytes=int(max_mb * 1024 * 1024),
            backupCount=backup, encoding="utf-8", delay=True
        )

    if compress:
        fh.namer = lambda name: f"{name}.gz"

        def _rotator(source, dest):
            with open(source, "rb") as sf, gzip.open(dest, "wb", compresslevel=5) as zf:
                shutil.copyfileobj(sf, zf)
            try:
                os.remove(source)
            except FileNotFoundError:
                pass

        fh.rotator = _rotator

    class ContextFilter(logging.Filter):
        def __init__(self, place, cuda):
            super().__init__()
            self.place = place or "-"
            self.cuda = cuda if cuda is not None else os.getenv("CUDA_VISIBLE_DEVICES", "-")

        def filter(self, record):
            record.place = getattr(record, "place", self.place)
            record.cuda = getattr(record, "cuda", self.cuda)
            return True

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [place=%(place)s cuda=%(cuda)s pid=%(process)d] %(name)s: %(message)s"
    )
    fh.setFormatter(fmt)
    fh.addFilter(ContextFilter(place, cuda))
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.addFilter(ContextFilter(place, cuda))
        root.addHandler(ch)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


def cleanup_old_logs(base_dir: str, keep_days: int = 30):
    """Read and clean .gz historical logs older than keep_days, only delete compressed rotated files"""
    now = time.time()
    base = pathlib.Path(base_dir)
    if not base.exists():
        return
    for p in base.rglob("*.gz"):
        try:
            if now - p.stat().st_mtime > keep_days * 86400:
                p.unlink()
        except FileNotFoundError:
            pass
