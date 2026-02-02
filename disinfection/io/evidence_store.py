# motiondetector/io/evidence_store.py
import os
import re
import cv2
import time
import random
import logging
import datetime

logger = logging.getLogger(__name__)


class EvidenceStore:
    def __init__(self, save_dir="./captures", place=None, save_interval=0.25):
        self.save_dir = save_dir
        self.place = place
        self.save_interval = save_interval
        self.last_save_time = time.time()
        self.last_save_id = ""
        self.persons_images = {}  # person_id -> [filepath...]

    def compress_frame(self, frame, max_width=640, max_height=480, quality=60):
        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        evidence = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', evidence, encode_param)
        evidence = cv2.imdecode(encimg, 1)
        return evidence

    def save_detection_image(self, frame, person_id, current_time_sec: float, save_count: int):
        new_save_id = f"{person_id}_{save_count}"
        if (current_time_sec - self.last_save_time >= self.save_interval) or (new_save_id != self.last_save_id):
            now = datetime.datetime.now()
            date_folder = now.strftime("%Y%m%d")
            save_dir = os.path.join(self.save_dir, self.place, date_folder) if self.place else os.path.join(self.save_dir, date_folder)
            os.makedirs(save_dir, exist_ok=True)

            micro = str(current_time_sec).split('.')[-1].ljust(6, '0')[:6]
            filename = f"person_{now.strftime('%Y%m%d_%H%M%S')}_{new_save_id}_{micro}.jpg"
            filepath = os.path.join(save_dir, filename)

            try:
                frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
                ok = cv2.imwrite(filepath, frame)
                if not ok:
                    logger.error("cv2.imwrite 返回 False: %s", filepath)
                    return None

                self.persons_images.setdefault(person_id, []).append(filepath)
                self.last_save_time = current_time_sec
                self.last_save_id = new_save_id
                logger.info("Saved person %s image: %s", person_id, filepath)
                return filepath
            except Exception as e:
                logger.error("Failed to save image: %s", e)
        return None

    def get_unqualified_images(self, person_id, limit=5, window_seconds=10):
        try:
            cached_files = self.persons_images.get(person_id, []).copy()
            results = []
            pattern = re.compile(r"^person_(\d{8})_(\d{6})_(\d+?)_(\d+?)_\d+\.jpg$")

            source_files = cached_files
            if not source_files:
                today = datetime.datetime.now().strftime("%Y%m%d")
                save_dir = os.path.join(self.save_dir, self.place, today) if self.place else os.path.join(self.save_dir, today)
                if not os.path.exists(save_dir):
                    return []
                source_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith("person_")]

            for path in source_files:
                fname = os.path.basename(path)
                m = pattern.match(fname)
                if not m:
                    continue
                date_str, time_str, pid_str, save_count_str = m.groups()
                if pid_str != str(person_id):
                    continue
                ts_dt = datetime.datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                results.append((ts_dt, int(save_count_str), path))

            if not results:
                return []

            results.sort(key=lambda x: (x[0], x[1]))
            latest_ts = results[-1][0]
            now = datetime.datetime.now()
            if (now - latest_ts).total_seconds() > 60:
                logger.info("Person %s newest image is older than 1 minute, ignoring", person_id)
                return []

            pool = [r for r in results if (latest_ts - r[0]).total_seconds() <= window_seconds] or results
            if len(pool) <= 3:
                return [p for _, _, p in pool][:limit]

            first = min(pool, key=lambda x: x[1])
            last = max(pool, key=lambda x: x[1])
            mid_candidates = [x for x in pool if first[1] < x[1] < last[1]]
            mid_sample = random.sample(mid_candidates, min(3, len(mid_candidates)))

            final = sorted([first] + mid_sample + [last], key=lambda x: (x[0], x[1]))
            return [p for _, _, p in final][:limit]

        except Exception as e:
            logger.error("Get %s recent images failed: %s", person_id, e)
            return []
