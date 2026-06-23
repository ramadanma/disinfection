# motiondetector/core/config.py
import argparse
import os
import re
from urllib.parse import urlparse


def _deep_get(d: dict, keys: str, default=None):
    cur = d or {}
    for k in keys.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_yaml_config(path: str) -> dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        import yaml  # pip install pyyaml
    except Exception as e:
        raise RuntimeError("missing dependency: pyyaml (pip install pyyaml)") from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_args():
    # 先解析 --config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.getenv("MOTIONDETECTOR_CONFIG", "configs/config.yaml"))
    pre_args, _ = pre.parse_known_args()
    cfg = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description="motiondetector (RTSP + YOLO Pose + Face)")
    parser.add_argument("--config", default=pre_args.config, help="YAML config path")

    # 基础
    parser.add_argument("--url", "-u",
                        default=_deep_get(cfg, "service.rtsp_url", os.getenv("RTSP_URL", "")) or
                                os.getenv("RTSP_URL", "rtsp://admin:password@ip:554/Streaming/Channels/101"))
    parser.add_argument("--place", "-p", default=_deep_get(cfg, "service.place", os.getenv("PLACE", "default_place")))
    parser.add_argument("--cuda", "-c", default=str(_deep_get(cfg, "service.cuda", os.getenv("CUDA", "0"))))
    parser.add_argument("--overprotocol", "-op",
                        default=_deep_get(cfg, "service.rtsp_protocol", os.getenv("RTSP_PROTOCOL", "tcp")))

    parser.add_argument("--device_id", "-id", default=_deep_get(cfg, "service.device_id", os.getenv("DEVICE_ID", "B009600001")))
    parser.add_argument("--report_count", "-rc",
                        default=int(_deep_get(cfg, "service.report_count", os.getenv("REPORT_COUNT", "5"))),
                        type=int)
    parser.add_argument("--required_time", "-rt",
                        default=float(_deep_get(cfg, "service.required_time", os.getenv("REQUIRED_TIME", "10.0"))),
                        type=float,
                        help="Seconds a person must stay in the pool (default: 10.0)")

    parser.add_argument("--pool", default=_deep_get(cfg, "service.pool_npy", os.getenv("POOL_NPY", "pool.npy")))
    parser.add_argument("--road", default=_deep_get(cfg, "service.road_npy", os.getenv("ROAD_NPY", "road.npy")))

    parser.add_argument("--known_dir", default=_deep_get(cfg, "service.known_dir", os.getenv("KNOWN_DIR", "known")))
    parser.add_argument("--models_root", default=_deep_get(cfg, "service.models_root", os.getenv("INSIGHTFACE_ROOT", "/home/motiondetector/.insightface")))

    parser.add_argument("--redis_host", default=_deep_get(cfg, "service.redis.host", os.getenv("REDIS_HOST", "127.0.0.1")))
    parser.add_argument("--redis_port", default=int(_deep_get(cfg, "service.redis.port", os.getenv("REDIS_PORT", "6379"))))
    parser.add_argument("--redis_queue", default=_deep_get(cfg, "service.redis.queue", os.getenv("REDIS_QUEUE", "downloaded")))

    # debug
    parser.add_argument("--draw_skeleton", action="store_true", default=bool(_deep_get(cfg, "service.debug.draw_skeleton", False)))
    parser.add_argument("--show_gui", action="store_true", default=bool(_deep_get(cfg, "service.debug.show_gui", False)))

    # YOLO Pose（关键点）配置：YAML 默认 + CLI 可覆盖
    parser.add_argument("--yolo_model", default=_deep_get(cfg, "service.yolo.model", "yolo11n-pose.pt"))
    parser.add_argument("--yolo_conf", type=float, default=float(_deep_get(cfg, "service.yolo.conf", 0.5)))
    parser.add_argument("--yolo_iou", type=float, default=float(_deep_get(cfg, "service.yolo.iou", 0.5)))
    parser.add_argument("--yolo_imgsz", type=int, default=int(_deep_get(cfg, "service.yolo.imgsz", 1280)))
    parser.add_argument("--yolo_tracker", default=_deep_get(cfg, "service.yolo.tracker", "bytetrack.yaml"))
    parser.add_argument("--yolo_max_det", type=int, default=int(_deep_get(cfg, "service.yolo.max_det", 50)))
    parser.add_argument("--yolo_persist", action="store_true", default=bool(_deep_get(cfg, "service.yolo.persist", True)))
    parser.add_argument("--yolo_agnostic_nms", action="store_true", default=bool(_deep_get(cfg, "service.yolo.agnostic_nms", True)))

    args = parser.parse_args()
    args._yaml_cfg = cfg
    return args


def extract_ip_from_rtsp(url: str):
    try:
        parsed = urlparse(url)
        if '@' in parsed.netloc:
            _, host_part = parsed.netloc.split('@', 1)
            return host_part.split(':')[0]
        return parsed.hostname
    except Exception:
        ip_match = re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url or "")
        return ip_match.group(0) if ip_match else None
