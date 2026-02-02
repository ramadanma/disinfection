# motiondetector/video/rtsp_reader.py
import time
import queue
import threading
import logging

logger = logging.getLogger(__name__)


class RtspReader:
    """
    put (timestamp_ms, bgr_image) into frame_q
    """
    def __init__(self, rtsp_url: str, protocol: str, frame_q: queue.Queue, stop_evt: threading.Event):
        self.rtsp_url = rtsp_url
        self.protocol = (protocol or "tcp").lower()
        self.frame_q = frame_q
        self.stop_evt = stop_evt
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def join(self, timeout=None):
        if self.thread:
            self.thread.join(timeout=timeout)

    def _run(self):
        import av
        av.logging.set_level(av.logging.ERROR)

        if self.protocol == "udp":
            opts = {
                "rtsp_transport": "udp",
                "fflags": "nobuffer+discardcorrupt",
                "buffer_size": "4194304",
                "max_delay": "500000",
                "probesize": "4M",
                "analyzeduration": "4M",
                "stimeout": "5000000",
                "rw_timeout": "5000000",
                "reorder_queue_size": "0",
            }
        else:
            opts = {
                "rtsp_transport": "tcp",
                "fflags": "nobuffer+discardcorrupt",
                "buffer_size": "4194304",
                "max_delay": "500000",
                "probesize": "4M",
                "analyzeduration": "4M",
                "stimeout": "5000000",
                "rw_timeout": "5000000",
                "rtsp_flags": "prefer_tcp",
                "reorder_queue_size": "0",
            }

        # last_ms 需要跨 packet 维持，否则会出现倒序帧
        last_ms = None

        while not self.stop_evt.is_set():
            try:
                container = av.open(self.rtsp_url, options=opts)
                video = container.streams.video[0]
                try:
                    video.thread_count = 1
                except Exception:
                    pass

                tb = video.time_base

                for packet in container.demux(video):
                    if self.stop_evt.is_set():
                        break
                    if (packet.dts is None and packet.pts is None) or getattr(packet, "is_corrupt", False):
                        continue

                    for frame in packet.decode():
                        if self.stop_evt.is_set():
                            break
                        if frame is None:
                            continue

                        pts = getattr(frame, "best_effort_timestamp", None)
                        if pts is None:
                            pts = frame.pts

                        if pts is not None and tb is not None:
                            ts_ms = int(pts * tb * 1000)
                        else:
                            t = getattr(frame, "time", None)
                            ts_ms = int(t * 1000) if t is not None else int(time.time() * 1000)

                        if last_ms is not None and ts_ms <= last_ms:
                            continue
                        last_ms = ts_ms

                        img = frame.to_ndarray(format="bgr24").copy()

                        try:
                            self.frame_q.put_nowait((ts_ms, img))
                        except queue.Full:
                            try:
                                self.frame_q.get_nowait()
                            except queue.Empty:
                                pass
                            self.frame_q.put_nowait((ts_ms, img))

                container.close()

            except Exception as e:
                logger.error("Failed to read RTSP stream: %s", e)
                time.sleep(1)
