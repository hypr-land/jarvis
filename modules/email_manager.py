from __future__ import annotations

"""Email Manager
~~~~~~~~~~~~~~~~
Utility class for sending plain-text emails via SMTP. Credentials and host
information are read from environment variables so that we do not need to
store secrets in the codebase or config files.

Required environment variables
------------------------------
EMAIL_SMTP_SERVER   – hostname of the smtp server (e.g. "smtp.gmail.com")
EMAIL_SMTP_PORT     – port number as int (usually 465 for SSL, 587 for STARTTLS)
EMAIL_USERNAME      – username / email address used for authentication
EMAIL_PASSWORD      – password or application-specific password
EMAIL_FROM          – optional "from" address (defaults to EMAIL_USERNAME)
"""

from typing import Dict, Any
import os
import smtplib
import ssl
from email.message import EmailMessage


class EmailManager:
    """Simple wrapper around *smtplib* for sending emails.

    Example
    -------
    >>> em = EmailManager()
    >>> em.send_email("alice@example.com", "Test", "Body")
    {'success': True}
    """

    def __init__(self) -> None:
        # Read settings from environment variables
        self.smtp_server = os.getenv("EMAIL_SMTP_SERVER")
        self.smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "465"))
        self.username = os.getenv("EMAIL_USERNAME")
        self.password = os.getenv("EMAIL_PASSWORD")
        self.from_addr = os.getenv("EMAIL_FROM", self.username)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def is_configured(self) -> bool:
        """Return *True* if all required env vars are present."""
        return bool(self.smtp_server and self.username and self.password)

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, Any]:
        """Send a plain-text e-mail.

        Parameters
        ----------
        to: str
            Recipient e-mail address.
        subject: str
            Subject line.
        body: str
            Plain-text body.
        """
        if not self.is_configured():
            return {
                "success": False,
                "error": "E-mail credentials are not configured. Set EMAIL_* environment variables",
            }

        msg = EmailMessage()
        msg["From"] = self.from_addr
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        try:
            # Prefer SMTPS (implicit SSL) if port is 465, otherwise STARTTLS
            if self.smtp_port == 465:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                    server.login(self.username, self.password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.ehlo()
                    server.starttls(context=ssl.create_default_context())
                    server.login(self.username, self.password)
                    server.send_message(msg)
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": str(exc)} 