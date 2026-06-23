# motiondetector/services/disinfection_service.py
import time
import logging
import signal
from pathlib import Path

from disinfection.core.config import parse_args, extract_ip_from_rtsp
from disinfection.core.logging_setup import setup_logging, cleanup_old_logs
from disinfection.analytics.pipeline import DisinfectionPipeline
from disinfection.video.rtsp_reader import RtspReader


def main():
    args = parse_args()
    cfg = getattr(args, "_yaml_cfg", {}) or {}

    setup_logging(
        place=args.place,
        cuda=str(args.cuda),
        rotate="time",
        when="midnight",
        backup=14,
        compress=True,
        console=True
    )

    project_root = Path(__file__).resolve().parents[2]
    cleanup_old_logs(base_dir=str(project_root / "logs"), keep_days=30)

    logger = logging.getLogger(__name__)
    url_ip = extract_ip_from_rtsp(args.url) or args.url

    yolo_cfg = {
        "model": args.yolo_model,
        "conf": args.yolo_conf,
        "iou": args.yolo_iou,
        "imgsz": args.yolo_imgsz,
        "tracker": args.yolo_tracker,
        "max_det": args.yolo_max_det,
        "persist": bool(args.yolo_persist),
        "agnostic_nms": bool(args.yolo_agnostic_nms),
        "classes": (cfg.get("service", {}).get("yolo", {}) or {}).get("classes", [0]),
    }

    pipeline = DisinfectionPipeline(
        cuda_id=int(args.cuda),
        place=args.place,
        pool_path=args.pool,
        road_path=args.road,
        protocol=args.overprotocol,
        device_id=args.device_id,
        url_ip=url_ip,
        known_dir=args.known_dir,
        models_root=args.models_root,
        report_count=args.report_count,
        required_time=args.required_time,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_queue=args.redis_queue,
        show_gui=bool(args.show_gui),
        draw_skeleton=bool(args.draw_skeleton),
        yolo_cfg=yolo_cfg,
        outputs_cfg=(cfg.get("service", {}).get("outputs", {}) or {}),
    )

    reader = RtspReader(
        rtsp_url=args.url,
        protocol=args.overprotocol,
        frame_q=pipeline.frame_q,
        stop_evt=pipeline.stop_evt,
    )

    def _handle_term(signum, frame):
        logger.info("received signal %s, stopping...", signum)
        pipeline.stop_evt.set()

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    pipeline.start()
    reader.start()

    try:
        while not pipeline.stop_evt.is_set():
            time.sleep(0.5)
    finally:
        pipeline.stop()
        reader.join(timeout=2)
        logger.info("motiondetector service stopped")


if __name__ == "__main__":
    main()
