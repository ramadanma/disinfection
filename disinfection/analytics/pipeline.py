# motiondetector/analytics/pipeline.py
import os
import time
import queue
import threading
import datetime
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import redis
import torch

from disinfection.analytics.roi import load_polygon_npy, load_roi_pts
from disinfection.analytics.yolo_pose import YoloPose
# Update the import path below if PersonStateMachine is defined elsewhere
from disinfection.analytics.state_machine import PersonStateMachine  # Ensure this path is correct

# If the above import fails, try one of the following alternatives:
# from disinfection.state_machine import PersonStateMachine
# from analytics.state_machine import PersonStateMachine
# from state_machine import PersonStateMachine
from disinfection.analytics import geometry

from disinfection.face.updater import reload_face_recognizer
from disinfection.io.evidence_store import EvidenceStore
from disinfection.io.reporter_email import EmailReporter

logger = logging.getLogger(__name__)


class DisinfectionPipeline:
    SKELETON_EDGES = [
        (0, 1), (0, 2),
        (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    def __init__(self,
                 cuda_id: int,
                 place: str,
                 pool_path: str,
                 road_path: str,
                 protocol: str,
                 device_id: str,
                 url_ip: str,
                 known_dir: str,
                 models_root: str,
                 report_count: int,
                 required_time: float = 10.0,
                 redis_host="127.0.0.1",
                 redis_port=6379,
                 redis_queue="downloaded",
                 show_gui=False,
                 draw_skeleton=False,
                 yolo_cfg=None,
                 outputs_cfg=None):

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        self.cuda_id = int(cuda_id)
        self.place = place
        self.protocol = protocol
        self.deviceID = device_id
        self.urlIp = url_ip
        self.report_count = report_count

        self.show_gui = bool(show_gui)
        self.draw_skeleton_enabled = bool(draw_skeleton)

        self.yolo_cfg = yolo_cfg or {}
        self.outputs_cfg = outputs_cfg or {}

        self.stop_evt = threading.Event()
        self._persons_lock = threading.Lock()
        self._identity_lock = threading.Lock()
        self._recognizer_lock = threading.Lock()

        self.frame_q = queue.Queue(maxsize=45)
        self.results_q = queue.Queue(maxsize=30)
        self.face_q = queue.Queue(maxsize=20)
        self.io_q = queue.Queue(maxsize=200)

        self.pool_region = load_polygon_npy(pool_path, name="pool")
        self.roi_pts = load_roi_pts(road_path)

        self.required_time = float(required_time)
        self.stable_threshold = 2
        self.keypoint_conf_threshold = 0.7
        self.keypoint_indices = {'left_ankle': 15, 'right_ankle': 16, 'nose': 0, 'left_ear': 3, 'right_ear': 4}

        self.stream_t0 = None
        self.last_results = None
        self.is_started = False
        self.identity_store = {}
        self.persons = {}

        # YOLO Pose from config
        self.detector = YoloPose(
            cuda_id=self.cuda_id,
            model=str(self.yolo_cfg.get("model", "yolo11n-pose.pt")),
            conf=float(self.yolo_cfg.get("conf", 0.5)),
        )

        self.evidence = EvidenceStore(save_dir="./captures", place=self.place, save_interval=0.25)

        # outputs: email only
        email_cfg = (self.outputs_cfg.get("email", {}) or {})
        self.email_reporter = None
        if bool(email_cfg.get("enabled", False)):
            self.email_reporter = EmailReporter(
                smtp_host=email_cfg.get("smtp_host"),
                smtp_port=email_cfg.get("smtp_port", 465),
                username=email_cfg.get("username"),
                password=email_cfg.get("password"),
                from_addr=email_cfg.get("from_addr"),
                to_addrs=email_cfg.get("to_addrs", []),
                use_ssl=bool(email_cfg.get("use_ssl", True)),
                use_starttls=bool(email_cfg.get("use_starttls", False)),
                subject_prefix=email_cfg.get("subject_prefix", "[motiondetector]"),
            )

        self.face_recognizer = reload_face_recognizer(
            old=None, ctx_id=self.cuda_id, known_dir=known_dir, models_root=models_root
        )

        self.state_machine = PersonStateMachine(
            pool_region=self.pool_region,
            stable_threshold=self.stable_threshold,
            required_time=self.required_time,
        )

        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
        self.REDIS_QUEUE = redis_queue

        self.font_path = self._get_chinese_font()

        logger.info("YOLO cfg: %s", self.yolo_cfg)
        try:
            logger.info("torch sees %d gpus: %s",
                        torch.cuda.device_count(),
                        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        except Exception:
            pass

    def _get_chinese_font(self):
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/microsoftyahei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
            "/usr/share/fonts/adobe-source-han-sans/OTF/SimplifiedChinese/SourceHanSansCN-Regular.otf"
        ]
        for p in font_paths:
            if os.path.exists(p):
                return p
        logger.warning("No fonts supporting Chinese were found, Chinese may not display properly.")
        return None

    def _put_chinese_text(self, img, text, position, size=20, color=(0, 255, 0)):
        if self.font_path:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.font_path, size)
            draw.text(position, text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, size / 10, color, 2)

    def draw_skeleton(self, frame, keypoints, conf_th=0.5,
                      point_color=(0, 255, 255), edge_color=(255, 0, 255)):
        if keypoints is None or len(keypoints) < 17:
            return frame
        for i in range(17):
            x, y, c = keypoints[i]
            if c >= conf_th:
                cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)
        for a, b in self.SKELETON_EDGES:
            xa, ya, ca = keypoints[a]
            xb, yb, cb = keypoints[b]
            if ca >= conf_th and cb >= conf_th:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), edge_color, 2)
        return frame

    def _get_foot_positions(self, keypoints):
        foot_positions = []
        la = keypoints[self.keypoint_indices['left_ankle']]
        ra = keypoints[self.keypoint_indices['right_ankle']]
        if la[2] > self.keypoint_conf_threshold:
            foot_positions.append((int(la[0]), int(la[1])))
        if ra[2] > self.keypoint_conf_threshold:
            foot_positions.append((int(ra[0]), int(ra[1])))
        return foot_positions

    def _get_nose_position(self, keypoints):
        nose = keypoints[self.keypoint_indices['nose']]
        if nose[2] > self.keypoint_conf_threshold:
            return (int(nose[0]), int(nose[1]))
        return ()

    def _get_ears_positions(self, keypoints):
        ears = []
        le = keypoints[self.keypoint_indices['left_ear']]
        re = keypoints[self.keypoint_indices['right_ear']]
        if le[2] > self.keypoint_conf_threshold:
            ears.append((int(le[0]), int(le[1])))
        if re[2] > self.keypoint_conf_threshold:
            ears.append((int(re[0]), int(re[1])))
        return ears

    def crop_face(self, frame, keypoints):
        nose = self._get_nose_position(keypoints)
        ears = self._get_ears_positions(keypoints)
        if not nose and not ears:
            return None
        xs, ys = [], []
        if nose:
            xs.append(nose[0]); ys.append(nose[1])
        for ex, ey in ears:
            xs.append(ex); ys.append(ey)
        x1, y1 = max(0, min(xs) - 30), max(0, min(ys) - 30)
        x2, y2 = min(frame.shape[1], max(xs) + 30), min(frame.shape[0], max(ys) + 30)
        return frame[y1:y2, x1:x2]

    def start(self):
        self.stop_evt.clear()
        self.detector.load()

        self.threads = [
            threading.Thread(target=self.inference_worker, daemon=True),
            threading.Thread(target=self.processing_worker, daemon=True),
            threading.Thread(target=self.face_worker, daemon=True),
            threading.Thread(target=self.io_worker, daemon=True),
            threading.Thread(target=self.download_worker, daemon=True),
            threading.Thread(target=self.clean_worker, daemon=True),
        ]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_evt.set()
        for t in getattr(self, "threads", []):
            t.join(timeout=1.5)

    def inference_worker(self):
        while not self.stop_evt.is_set():
            try:
                ts_ms, frame = self.frame_q.get(timeout=0.05)
            except queue.Empty:
                continue

            results = None
            try:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [self.roi_pts], 255)
                roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

                results = self.detector.track(
                    roi_frame,
                    device=f'cuda:{self.cuda_id}',
                    classes=self.yolo_cfg.get("classes", [0]),
                    conf=float(self.yolo_cfg.get("conf", 0.5)),
                    iou=float(self.yolo_cfg.get("iou", 0.5)),
                    imgsz=int(self.yolo_cfg.get("imgsz", 1280)),
                    agnostic_nms=bool(self.yolo_cfg.get("agnostic_nms", True)),
                    persist=bool(self.yolo_cfg.get("persist", True)),
                    tracker=str(self.yolo_cfg.get("tracker", "bytetrack.yaml")),
                    show=False,
                    verbose=False,
                    max_det=int(self.yolo_cfg.get("max_det", 50)),
                )
            except Exception as e:
                logger.warning("track failed, fallback predict: %s", e)
                try:
                    results = self.detector.predict(frame, imgsz=int(self.yolo_cfg.get("imgsz", 1280)))
                except Exception as e2:
                    logger.error("predict failed: %s", e2)
                    results = None

            self.last_results = results

            if self.results_q.full():
                try:
                    self.results_q.get_nowait()
                except queue.Empty:
                    pass
            self.results_q.put_nowait((ts_ms, frame, results))
            self.frame_q.task_done()

    def face_worker(self):
        while not self.stop_evt.is_set():
            try:
                person_id, face_img = self.face_q.get(timeout=0.05)
            except queue.Empty:
                continue

            with self._recognizer_lock:
                recognizer = self.face_recognizer  # snapshot reference; safe to use outside lock

            name, conf = recognizer.recognize(face_img, person_id)
            final_name = recognizer.update_identity(person_id, name, conf)

            with self._identity_lock:
                self.identity_store.setdefault(person_id, {"final_name": "unknown"})
                self.identity_store[person_id]["final_name"] = final_name

    def io_worker(self):
        while not self.stop_evt.is_set():
            try:
                task = self.io_q.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                upload = task.get('upload', False)
                save_count = task.get('save_count', 0)

                if not upload:
                    img = task.get('image')
                    person_id = task.get('person_id')
                    timestamp = task.get('timestamp', datetime.datetime.now())

                    self.evidence.save_detection_image(
                        img, person_id,
                        current_time_sec=timestamp.timestamp(),
                        save_count=save_count
                    )
                else:
                    person_id = task.get('person_id')
                    unqualified_limit = min(save_count, 5) if save_count > 0 else 5

                    images = self.evidence.get_unqualified_images(person_id, unqualified_limit)

                    with self._identity_lock:
                        identity = (self.identity_store.get(person_id, {}) or {}).get('final_name', 'unknown')

                    with self._persons_lock:
                        person = self.persons.get(person_id, {})
                        state = person.get('state', 'unknown')
                        total_duration = float(person.get('total_duration', 0.0))

                    if images and self.email_reporter is not None:
                        self.email_reporter.send_report(
                            images, person_id, identity, state, total_duration, self.required_time,
                            place=self.place
                        )

                        self.evidence.clear_person_images(person_id)
                        with self._persons_lock:
                            person_ref = self.persons.get(person_id)
                            if person_ref is not None:
                                person_ref['has_recorded'] = True
                                person_ref['should_record'] = False

            except Exception as e:
                logger.error("io_worker error: %s", e)
            finally:
                self.io_q.task_done()

    def download_worker(self):
        while not self.stop_evt.is_set():
            try:
                if self.face_q.empty():
                    redis_info = self.redis_client.blpop(self.REDIS_QUEUE, timeout=1)
                    if not redis_info:
                        continue
                    _, message = redis_info

                    if message in ('delete', 'update'):
                        logger.info("收到更新人脸库指令=%s，正在重加载人脸识别模块", message)
                        with self._recognizer_lock:
                            old = self.face_recognizer
                        new_recognizer = reload_face_recognizer(
                            old=old,
                            ctx_id=self.cuda_id,
                            known_dir=old.known_dir,
                            models_root=old.models_root,
                        )
                        with self._recognizer_lock:
                            self.face_recognizer = new_recognizer
                else:
                    time.sleep(0.05)
            except Exception as e:
                logger.error("download_worker error: %s", e)
                time.sleep(1)

    def clean_worker(self):
        while not self.stop_evt.is_set():
            time.sleep(1200)
            try:
                if self.face_q.empty() and self.io_q.empty() and len(self.persons) > 50:
                    logger.info("清理 persons 内存，占用过大")
                    with self._persons_lock:
                        self.persons.clear()
                    self.evidence.clear_all_images()
                    with self._identity_lock:
                        self.identity_store.clear()
                    with self._recognizer_lock:
                        recognizer = self.face_recognizer
                    recognizer.identity_history.clear()
            except Exception as e:
                logger.error("clean_worker error: %s", e)

    def processing_worker(self):
        if self.show_gui:
            cv2.namedWindow("disinfection pool", cv2.WINDOW_NORMAL)

        while not self.stop_evt.is_set():
            try:
                ts_ms, frame, results = self.results_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if self.stream_t0 is None:
                self.stream_t0 = time.time() - ts_ms / 1000.0
            current_time = self.stream_t0 + ts_ms / 1000.0

            if results is None:
                results = self.last_results
            if not isinstance(results, (list, tuple)):
                results = [results] if results is not None else []

            if not self.is_started:
                try:
                    self.io_q.put_nowait({
                        'image': frame,
                        'person_id': "###",
                        'timestamp': datetime.datetime.now(),
                        'save_count': 0,
                        'upload': False,
                    })
                    self.is_started = True
                except queue.Full:
                    logger.warning("IO queue is full, discarding startup save request")

            cv2.polylines(frame, self.pool_region, True, (0, 255, 0), 2)
            first_point = self.pool_region[0][0]
            frame = self._put_chinese_text(frame, "消毒池", (first_point[0], first_point[1] - 30), size=25, color=(0, 255, 0))

            for result in results:
                if not hasattr(result, 'boxes') or not hasattr(result, 'keypoints'):
                    continue
                if result.keypoints is None:
                    continue

                for box, kpts in zip(result.boxes, result.keypoints.data):
                    if box.id is None:
                        continue

                    person_id = int(box.id)
                    keypoints = kpts.cpu().numpy()

                    if self.draw_skeleton_enabled:
                        self.draw_skeleton(frame, keypoints, conf_th=0.5)

                    foot_positions = self._get_foot_positions(keypoints)
                    nose_position = self._get_nose_position(keypoints)

                    both_feet_in = False
                    if len(foot_positions) == 2:
                        both_feet_in = (geometry.is_point_in_region(foot_positions[0], self.pool_region) and
                                        geometry.is_point_in_region(foot_positions[1], self.pool_region))

                    with self._persons_lock:
                        person = self.state_machine.update(
                            self.persons,
                            self.evidence.persons_images,
                            person_id,
                            both_feet_in,
                            current_time,
                            foot_positions=foot_positions,
                            nose_position=nose_position
                        )
                        should_save = person.get('should_save_image', False)
                        unqualified_count = person.get('unqualified_count', 0)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    face_img = self.crop_face(frame, keypoints)
                    if face_img is None:
                        face_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

                    if face_img is not None and face_img.size > 0 and (not self.face_q.full()):
                        self.face_q.put_nowait((person_id, face_img))

                    if should_save:
                        evidence_img = self.evidence.compress_frame(frame)
                        try:
                            self.io_q.put_nowait({
                                'upload': False,
                                'image': evidence_img,
                                'person_id': person_id,
                                'timestamp': datetime.datetime.now(),
                                'save_count': unqualified_count,
                            })
                            with self._persons_lock:
                                p = self.persons.get(person_id)
                                if p is not None:
                                    p['should_save_image'] = False
                        except queue.Full:
                            logger.warning("IO queue is full, discarding save request person=%s", person_id)

                    with self._persons_lock:
                        p = self.persons.get(person_id)
                        if p is not None and p.get('should_record', False) and not p.get('has_recorded', False):
                            try:
                                self.io_q.put_nowait({
                                    'upload': True,
                                    'person_id': person_id,
                                    'save_count': p.get('unqualified_count', 0),
                                })
                                p['has_recorded'] = True
                                p['should_record'] = False
                            except queue.Full:
                                logger.warning("IO queue is full, discarding record request person=%s", person_id)

            self.results_q.task_done()
