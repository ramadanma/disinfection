# motiondetector/io/reporter_email.py
import os
import ssl
import logging
import smtplib
from typing import List, Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

logger = logging.getLogger(__name__)


class EmailReporter:
    def __init__(self,
                 smtp_host: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 from_addr: str,
                 to_addrs: List[str],
                 use_ssl: bool = True,
                 use_starttls: bool = False,
                 subject_prefix: str = "[motiondetector]"):
        self.smtp_host = smtp_host
        self.smtp_port = int(smtp_port)
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs or []
        self.use_ssl = bool(use_ssl)
        self.use_starttls = bool(use_starttls)
        self.subject_prefix = subject_prefix or "[motiondetector]"

    def send_report(self,
                    filepaths: List[str],
                    person_id,
                    identity: str,
                    state: str,
                    total_duration: float,
                    required_time: float,
                    place: Optional[str] = None):
        if not self.to_addrs:
            logger.warning("EmailReporter: empty to_addrs, skip sending")
            return

        subject = f"{self.subject_prefix} {place or ''} person={person_id} state={state}".strip()

        body = (
            f"place: {place}\n"
            f"person_id: {person_id}\n"
            f"identity: {identity}\n"
            f"state: {state}\n"
            f"duration_sec: {total_duration:.2f}\n"
            f"required_sec: {required_time:.2f}\n"
            f"images: {len(filepaths)}\n"
        )

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # attach images (up to 5)
        for p in (filepaths or [])[:5]:
            if not os.path.exists(p):
                continue
            try:
                with open(p, "rb") as f:
                    part = MIMEApplication(f.read(), Name=os.path.basename(p))
                part["Content-Disposition"] = f'attachment; filename="{os.path.basename(p)}"'
                msg.attach(part)
            except Exception as e:
                logger.error("EmailReporter attach failed: %s %s", p, e)

        try:
            if self.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context, timeout=20) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as server:
                    server.ehlo()
                    if self.use_starttls:
                        server.starttls(context=ssl.create_default_context())
                        server.ehlo()
                    server.login(self.username, self.password)
                    server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            logger.info("EmailReporter sent: to=%s subject=%s", self.to_addrs, subject)
        except Exception as e:
            logger.error("EmailReporter send failed: %s", e)
