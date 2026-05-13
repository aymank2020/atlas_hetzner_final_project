import os
import sys
import time
import traceback
from playwright.sync_api import sync_playwright


def _gemini_sign_in_required(page) -> bool:
    try:
        url = str(page.url or "")
    except Exception:
        url = ""
    try:
        title = str(page.title() or "")
    except Exception:
        title = ""
    try:
        body = str(page.locator("body").inner_text(timeout=2000) or "")
    except Exception:
        body = ""
    lower = body.lower()
    return (
        "accounts.google.com" in url
        or "sign in" in title.lower()
        or "sign in gemini" in lower
        or "get access to all gemini models" in lower
        or ("meet gemini, your personal ai assistant" in lower and "sign in" in lower)
    )


def _cleanup_probe(browser, context, page, *, owns_context=False):
    try:
        if page is not None:
            page.close()
    except Exception:
        pass
    if owns_context:
        try:
            if context is not None:
                context.close()
        except Exception:
            pass
    # Never call browser.close() on a shared CDP-attached browser here.
    # Exiting sync_playwright() is enough to drop this probe connection safely.


def _locator_is_visible(locator, *, timeout=1000):
    try:
        return bool(locator.is_visible(timeout=timeout))
    except Exception:
        return False


def _handle_google_account_chooser(page, email):
    try:
        url = str(page.url or "")
    except Exception:
        url = ""
    if "accounts.google.com" not in url:
        return False

    email_input = page.locator("input[type='email']")
    if _locator_is_visible(email_input, timeout=800):
        return False

    email_text = str(email or "").strip()
    if email_text:
        matching_account = page.locator(f"text='{email_text}'").first
        if _locator_is_visible(matching_account, timeout=1200):
            matching_account.click()
            page.wait_for_timeout(1200)
            return True

    use_another_account = page.locator("text='Use another account'").first
    if _locator_is_visible(use_another_account, timeout=1200):
        use_another_account.click()
        page.wait_for_timeout(1200)
        return True

    return False


def do_auto_login(page, email, password):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] Attempting automated login...")
    try:
        if "gemini.google.com" in page.url and _gemini_sign_in_required(page):
            # Click the sign in button on the Gemini splash page
            sign_in_btn = page.locator("text='Sign in'").first
            if _locator_is_visible(sign_in_btn, timeout=5000):
                sign_in_btn.click()
            page.wait_for_load_state("networkidle")
            
        if "accounts.google.com" in page.url:
            _handle_google_account_chooser(page, email)

            # Type email
            email_input = page.locator("input[name='identifier'], input[type='email']").first
            if _locator_is_visible(email_input, timeout=2000):
                email_input.wait_for(state="visible", timeout=10000)
                email_input.fill(email)
                page.keyboard.press("Enter")
            
            # Wait for password input to appear
            password_input = page.locator("input[name='Passwd'], input[aria-label='Enter your password']").first
            password_input.wait_for(state="visible", timeout=15000)
            
            # Additional small delay for Google's transitions
            time.sleep(1)
            
            # Type password
            password_input.fill(password)
            page.keyboard.press("Enter")
            
            # Wait for navigation back to Gemini or an error
            page.wait_for_load_state("networkidle", timeout=20000)
            time.sleep(3) # Give it a moment to land
            
            if "gemini.google.com" in page.url or "Gemini" in page.title():
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] ✅ Auto-login succeeded.")
                return True
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] ⚠️ Auto-login stuck. Might be asking for 2FA/Captcha.")
                return False
        return False
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] ❌ Auto-login error: {e}")
        return False

def monitor():
    # Example GEMINI_ACCOUNTS in .env:  email1:pass1,email2:pass2 
    accounts_str = os.environ.get("GEMINI_ACCOUNTS", "").strip()
    accounts = []
    if accounts_str:
        for pair in accounts_str.split(','):
            parts = pair.split(':', 1)
            if len(parts) == 2:
                accounts.append((parts[0].strip(), parts[1].strip()))
    else:
        email = os.environ.get("GEMINI_EMAIL", "").strip()
        password = os.environ.get("GEMINI_PASSWORD", "").strip()
        if email and password:
            accounts.append((email, password))
            
    interval_seconds = int(os.environ.get("HEARTBEAT_INTERVAL_SEC", 600))
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] Starting monitor. Interval: {interval_seconds}s. Total Accounts: {len(accounts)}")
    
    while True:
        try:
            with sync_playwright() as p:
                browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222", timeout=10000)
                owns_context = not bool(browser.contexts)
                context = browser.contexts[0] if browser.contexts else browser.new_context()
                page = context.new_page()
                
                for idx, (email, password) in enumerate(accounts):
                    # Check the status of each authuser index
                    page.goto(f"https://gemini.google.com/app?authuser={idx}", wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(5000)
                    
                    if _gemini_sign_in_required(page):
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] ❌ [AuthUser={idx}] ({email}) Status: LOGGED OUT. Attempting login...")
                        # Safely trigger login for a specific additional account
                        page.goto("https://accounts.google.com/AddSession?hl=en&continue=https://gemini.google.com/app", wait_until="domcontentloaded")
                        page.wait_for_timeout(3000)
                        success = do_auto_login(page, email, password)
                        if not success:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] 🚨 Manual Intervention Required for {email} via VNC.")
                    else:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] ✅ [AuthUser={idx}] ({email}) Status: LOGGED IN")
                
                _cleanup_probe(browser, context, page, owns_context=owns_context)
        except BaseException as e:
            # We catch BaseException to ensure transient CDP connection errors don't crash the heartbeat daemon
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [HEARTBEAT] ⚠️ Failed to fetch status. (CDP down or timeout): {e}")

        # Sleep until the next heartbeat check
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # Add a short startup delay to let the primary script launch services if needed
    time.sleep(10)
    monitor()
