# motiondetector/services/face_library_service.py
import time
import logging
import redis

from disinfection.core.config import parse_args
from disinfection.core.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    setup_logging(place=f"{args.place}_face_library", cuda=str(args.cuda), console=True)

    r = redis.StrictRedis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    q = args.redis_queue

    logger.info("face_library_service started. redis=%s:%s queue=%s", args.redis_host, args.redis_port, q)
    logger.info("最小版本：每30秒发送一次 update 指令（你可改为触发式）")

    while True:
        time.sleep(30)
        r.rpush(q, "update")
        logger.info("sent: update")


if __name__ == "__main__":
    main()
