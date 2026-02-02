# motiondetector/face/recognizer.py
import os
import logging
import numpy as np
import cv2
import insightface

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(self, known_dir="../known", threshold=0.35, history_len=20, ctx_id=0, models_root="../.insightface"):
        self.known_dir = known_dir
        self.threshold = threshold
        self.history_len = history_len
        self.models_root = models_root

        self.recognizer = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=self.models_root,
            providers=[('CUDAExecutionProvider', {'device_id': int(ctx_id)}), 'CPUExecutionProvider']
        )
        self.recognizer.prepare(ctx_id=ctx_id)

        self.known_embeddings, self.known_labels = self.load_known_faces()
        self.identity_history = {}  # {person_id: [(name, conf), ...]}

    def load_known_faces(self):
        embeddings, labels = [], []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if not os.path.exists(self.known_dir):
            logger.warning("known_dir 不存在: %s", self.known_dir)
            return np.zeros((1, 512)), ["unknown"]

        for root, _, files in os.walk(self.known_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in valid_exts:
                    continue

                path = os.path.join(root, fn)
                name = os.path.splitext(fn)[0]

                try:
                    img = cv2.imread(path)
                    if img is None:
                        logger.warning("can't read the image file: %s", path)
                        continue
                    faces = self.recognizer.get(img)
                    if not faces:
                        logger.warning("No faces detected in image: %s", path)
                        continue
                    embeddings.append(faces[0].embedding)
                    labels.append(name)
                except Exception as e:
                    logger.error("Error processing image %s: %s", path, e)

        if len(embeddings) == 0:
            logger.warning("No face data loaded, using default unknown.")
            embeddings = np.zeros((1, 512))
            labels = ["unknown"]

        return np.array(embeddings), labels

    def recognize(self, face_img, person_id):
        faces = self.recognizer.get(face_img)
        if len(faces) == 0:
            return "unknown", 0.0
        emb = faces[0].embedding

        sims = np.dot(self.known_embeddings, emb) / (
            np.linalg.norm(self.known_embeddings, axis=1) * np.linalg.norm(emb) + 1e-6
        )
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score > self.threshold:
            name = self.known_labels[best_idx]
            logger.info("Person %s recognized as %s (score=%.3f)", person_id, name, best_score)
            return name, best_score
        return "unknown", best_score

    def update_identity(self, person_id, name, conf):
        if person_id not in self.identity_history:
            self.identity_history[person_id] = []
        history = self.identity_history[person_id]

        history.append((name, conf))
        if len(history) > self.history_len:
            history.pop(0)

        filtered_names = [n for n, _ in history if n != "unknown"]
        if len(filtered_names) == 0:
            return "unknown"

        base_names = [n.split("_")[0] for n in filtered_names]
        if len(set(base_names)) == 1:
            return base_names[0]

        scores = {}
        for n, c in history:
            if n == "unknown":
                continue
            base = n.split("_")[0]
            scores.setdefault(base, []).append(c)

        avg_scores = {k: float(np.mean(v)) for k, v in scores.items()}
        return max(avg_scores, key=avg_scores.get)
