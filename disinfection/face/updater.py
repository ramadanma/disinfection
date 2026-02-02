# motiondetector/face/updater.py
import logging
from disinfection.face.recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


def reload_face_recognizer(old: FaceRecognizer, ctx_id: int, known_dir: str, models_root: str):
    logger.info("Reload face recognizer block: known_dir=%s ctx_id=%s", known_dir, ctx_id)
    return FaceRecognizer(ctx_id=ctx_id, known_dir=known_dir, models_root=models_root)
