"""Quick OTP fetcher: connects to Gmail IMAP and extracts the LATEST Atlas verification code."""
import imaplib
import email
import re
import os
import sys
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

GMAIL_USER = os.getenv("GMAIL_USER", "").strip().strip('"')
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").strip().strip('"')

print(f"[OTP] Connecting as {GMAIL_USER}...")
imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
imap.login(GMAIL_USER, GMAIL_APP_PASSWORD)
imap.select("INBOX")

# Search for recent Atlas verification emails
status, data = imap.search(None, 'ALL')
if status != "OK" or not data[0]:
    print("NO_EMAILS")
    sys.exit(1)

uids = data[0].split()
# Check last 10 emails for Atlas OTP
best_code = None
best_time = None

for uid in uids[-10:]:
    status, msg_data = imap.fetch(uid, "(RFC822)")
    if status != "OK" or msg_data[0] is None:
        continue
    raw = msg_data[0][1]
    msg = email.message_from_bytes(raw)
    
    subject = str(msg.get("Subject", ""))
    from_addr = str(msg.get("From", ""))
    date_str = msg.get("Date", "")
    
    # Only look at Atlas-related emails
    if "atlas" not in from_addr.lower() and "verification" not in subject.lower() and "code" not in subject.lower():
        continue
    
    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct in ("text/plain", "text/html"):
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="replace")
    
    # Find 6-digit codes (exclude 000000)
    codes = re.findall(r'\b(\d{6})\b', body)
    codes = [c for c in codes if c != "000000"]
    
    if codes:
        try:
            msg_time = parsedate_to_datetime(date_str)
        except Exception:
            msg_time = datetime.now(timezone.utc)
        
        if best_time is None or msg_time > best_time:
            best_code = codes[-1]
            best_time = msg_time
            print(f"  Found code {best_code} from {from_addr} at {date_str}")

imap.logout()

if best_code:
    print(f"\nOTP_CODE={best_code}")
else:
    print("\nNO_VALID_CODE_FOUND")
    # Try showing last few email subjects for debugging
    print("(Searched last 10 emails, none had Atlas verification codes)")
