import os
from pathlib import Path

from playwright.sync_api import sync_playwright

state = Path(".state/gemini_chat_storage_state.json")
state.parent.mkdir(parents=True, exist_ok=True)
chat_url = os.environ.get("GEMINI_CHAT_WEB_URL", "").strip() or "https://gemini.google.com/app/b3006ba9f325b55c"
user_data_dir_raw = os.environ.get("GEMINI_CHAT_USER_DATA_DIR", "").strip() or ".state/gemini_chat_user_data"
user_data_dir = Path(user_data_dir_raw)
if not user_data_dir.is_absolute():
    user_data_dir = (Path.cwd() / user_data_dir).resolve()
user_data_dir.mkdir(parents=True, exist_ok=True)
chrome_channel = os.environ.get("GEMINI_CHAT_CHROME_CHANNEL", "").strip() or "chrome"
cdp_url = os.environ.get("GEMINI_CHAT_CONNECT_OVER_CDP_URL", "").strip()
launch_args = [
    "--disable-blink-features=AutomationControlled",
]
stealth_script = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = window.chrome || { runtime: {} };
"""

with sync_playwright() as p:
    browser = None
    context = None
    if cdp_url:
        browser = p.chromium.connect_over_cdp(cdp_url)
        if browser.contexts:
            context = browser.contexts[0]
        else:
            context = browser.new_context()
    else:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            channel=chrome_channel,
            headless=False,
            no_viewport=True,
            ignore_default_args=["--enable-automation"],
            args=launch_args,
        )
    context.add_init_script(stealth_script)
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(chat_url, wait_until="domcontentloaded")
    print("Sign in to Gmail/Gemini in the opened browser window.")
    if cdp_url:
        print("mode: attach-to-existing-chrome")
        print("cdp_url:", cdp_url)
    else:
        print("mode: persistent-chrome-profile")
    input("Press Enter here after login to save the storage state...")
    context.storage_state(path=str(state))
    if browser is not None:
        browser.close()
    else:
        context.close()

print("saved:", state.resolve())
print("url:", chat_url)
print("user_data_dir:", user_data_dir)
print("channel:", chrome_channel)
