"""Capture start2.json from the live KanColle game using Playwright + Edge.

Why this approach:
  * Logging in to DMM (possible captcha/2FA) is done by a real browser you can
    see. The script only automates the tedious parts (typing credentials,
    clicking "game start") and otherwise waits.
  * It never *replays* any auth request itself -- it passively watches the
    network traffic of the real page and grabs the api_start2/getData
    response the game makes on its own. So there is nothing to detect and
    nothing that breaks when DMM changes its token handling.

Flow:
  1. Launch Edge with a persistent profile (so you usually stay logged in).
  2. Open the DMM play page.
  3. **Region-block workaround**: from a non-JP IP the play page redirects to
     special.dmm.com/not-available-in-your-region. The trick (same as the
     GotoBrowser Android app) is to set DMM's "domestic" marker cookies
     (ckcy=1 and ckcy_remedied_check) and bounce back to the play page.
     No VPN needed, and the gadget host (w00g.kancolle-server.com) then loads
     normally.
  4. If a login form is visible and credentials are available, fill + submit.
  5. Wait for you (or auto-click) to reach the game.
  6. The moment api_start2/getData is seen, save it as start2.json and close.

Credentials (optional -- only needed if the profile isn't already logged in):
  Set environment variables:
      KC_DMM_USER   your DMM email/login
      KC_DMM_PASS   your DMM password
  or create ./credentials.json:
      {"user": "...", "pass": "..."}
  If neither is present, the script simply lets you type them by hand.

Usage:
    python capture.py
    python capture.py --out start2.json
    python capture.py --keep-open      # don't close the browser when done
    python capture.py --headless       # not recommended (can't solve challenges)
    python capture.py --no-cookie-fix  # disable the ckcy region bypass (if you use a VPN)
"""

import argparse
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "start2.json")
PROFILE_DIR = os.path.join(HERE, ".browser-profile")

DMM_PLAY_URL = "https://play.games.dmm.com/game/kancolle"
# The API endpoint whose response we want (matched as a substring).
START2_URL_MARKER = "api_start2/getData"

# --- region-block workaround (same idea as the GotoBrowser Android app) -----
# The play page redirects non-JP IPs to one of these; when we see one we set
# DMM's domestic cookies and go back to the play page.
REGION_BLOCK_MARKERS = (
    "special.dmm.com/not-available-in-your-region",
    "www.dmm.com/netgame/foreign",
)

# DMM's "I am a domestic user" cookies. ckcy=1 is the important one; the
# "remedied_check" cookie short-circuits DMM's client-side re-check.
# (Values taken from GotoBrowser Constants.DMM_COOKIE.)
COOKIE_DOMAIN = ".dmm.com"
COOKIE_MAX_AGE_S = 365 * 5 * 24 * 60 * 60  # 5 years, as in GotoBrowser
CK_CY = "1"
CK_CY_REMEDIED_CHECK = "ec_mrnhbtk"


def _load_credentials():
    """Return (user, password) from env vars or ./credentials.json, else (None, None)."""
    user = os.environ.get("KC_DMM_USER")
    pw = os.environ.get("KC_DMM_PASS")
    if user and pw:
        return user, pw

    cred_file = os.path.join(HERE, "credentials.json")
    if os.path.exists(cred_file):
        try:
            with open(cred_file, encoding="utf-8") as fh:
                data = json.load(fh)
            return data.get("user"), data.get("pass")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not read credentials.json: {exc}")
    return None, None


def _strip_svdata(text):
    """The API wraps JSON as: svdata={...}. Return just the JSON object."""
    text = text.strip()
    if text.startswith("svdata="):
        text = text[len("svdata="):]
    return text


def _is_region_blocked(url):
    return any(marker in url for marker in REGION_BLOCK_MARKERS)


def _set_domestic_cookies(ctx):
    """Set DMM's domestic-user marker cookies (defeats the region gate)."""
    expires = time.time() + COOKIE_MAX_AGE_S
    ctx.add_cookies([
        {"name": "ckcy", "value": CK_CY, "domain": COOKIE_DOMAIN, "path": "/",
         "expires": expires},
        {"name": "ckcy_remedied_check", "value": CK_CY_REMEDIED_CHECK,
         "domain": COOKIE_DOMAIN, "path": "/", "expires": expires},
    ])


def _try_autofill_login(page, user, pw):
    """Best-effort autofill of the DMM login form. Returns True if submitted.

    DMM's markup changes over time, so we try several common selectors and
    bail out quietly if none match -- in that case the user logs in manually.
    """
    if not (user and pw):
        return False
    try:
        # email/login field
        email_sel = None
        for sel in (
            "input[type=email]",
            "input[name=login_id]",
            "input[name=email]",
            "#login_id",
        ):
            if page.locator(sel).count() > 0:
                email_sel = sel
                break
        pw_sel = None
        for sel in (
            "input[type=password]",
            "input[name=password]",
            "#password",
        ):
            if page.locator(sel).count() > 0:
                pw_sel = sel
                break

        if not (email_sel and pw_sel):
            return False

        print("  autofilling DMM credentials...")
        page.fill(email_sel, user)
        page.fill(pw_sel, pw)

        # submit button
        for sel in (
            "button[type=submit]",
            "input[type=submit]",
            "button:has-text('ログイン')",
            "button:has-text('Login')",
        ):
            if page.locator(sel).count() > 0:
                page.click(sel)
                return True
        page.press(pw_sel, "Enter")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  autofill skipped ({exc}); please log in manually")
        return False


def _try_click_game_start(page):
    """Best-effort click of the 'game start' button on the play page."""
    for sel in (
        "a:has-text('ゲームスタート')",
        "button:has-text('ゲームスタート')",
        "a:has-text('ゲームをはじめる')",
        "button:has-text('ゲームをはじめる')",
        "a:has-text('Game Start')",
        "button:has-text('Game Start')",
        "a:has-text('ゲーム開始')",
    ):
        try:
            if page.locator(sel).count() > 0:
                page.click(sel, timeout=2000)
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


def capture(out_path=DEFAULT_OUT, keep_open=False, headless=False, timeout_s=600,
            use_cookie_fix=True):
    """Open Edge, let the user reach the game, capture api_start2/getData."""
    user, pw = _load_credentials()
    if user and pw:
        print("credentials loaded (will autofill if a login form appears)")
    else:
        print("no credentials configured; please log in by hand when prompted")

    captured = {}

    with sync_playwright() as p:
        try:
            ctx = p.chromium.launch_persistent_context(
                PROFILE_DIR,
                channel="msedge",
                headless=headless,
                viewport={"width": 1280, "height": 800},
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "existing browser session" in msg or "already in use" in msg:
                print("\n! The browser profile is already open in another Edge "
                      "window.")
                print("  Close that window (or run: "
                      "Stop-Process -Name msedge -Force) and try again.")
                return False
            raise
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        def on_response(resp):
            if START2_URL_MARKER in resp.url and "body" not in captured:
                try:
                    captured["body"] = resp.text()
                    print(f"  captured {START2_URL_MARKER} from {resp.url}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! could not read response body: {exc}")

        page.on("response", on_response)

        print(f"opening {DMM_PLAY_URL}")
        page.goto(DMM_PLAY_URL, wait_until="domcontentloaded", timeout=60000)

        def handle_region_block():
            """If we're on the region page, set cookies and bounce back."""
            if not (use_cookie_fix and _is_region_blocked(page.url)):
                return False
            print("  region block detected -> setting ckcy cookies and retrying")
            _set_domestic_cookies(ctx)
            try:
                page.goto(DMM_PLAY_URL, wait_until="domcontentloaded",
                          timeout=60000)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! retry navigation failed: {exc}")
            page.wait_for_timeout(3000)
            return True

        # Handle the region block that fires immediately on first load, then
        # give a login form a moment to appear and try to autofill it.
        handle_region_block()
        page.wait_for_timeout(3000)
        _try_autofill_login(page, user, pw)

        print("waiting for you to reach the game (and the start2 request)...")
        print("  -> finish any login / captcha / game-start by hand if needed")

        waited = 0
        clicked_start = False
        while "body" not in captured and waited < timeout_s:
            page.wait_for_timeout(1000)
            waited += 1
            # the region block can also reappear after the login redirect
            if waited % 3 == 1:
                handle_region_block()
            # periodically try to click "game start" if it's there
            if not clicked_start and waited % 3 == 0:
                clicked_start = _try_click_game_start(page)

        body = captured.get("body")
        if not body:
            print("timed out before start2 was captured")
            if keep_open:
                input("press Enter to close the browser...")
            ctx.close()
            return False

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(_strip_svdata(body))
        print(f"saved -> {out_path}")

        if keep_open:
            input("press Enter to close the browser...")
        ctx.close()
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT, help="where to write start2.json")
    ap.add_argument("--keep-open", action="store_true",
                    help="leave the browser open until you press Enter")
    ap.add_argument("--headless", action="store_true",
                    help="run without a window (you can't solve challenges then)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds to wait for start2 before giving up")
    ap.add_argument("--no-cookie-fix", action="store_true",
                    help="disable the ckcy region bypass (use if you already have a VPN)")
    args = ap.parse_args(argv)

    ok = capture(out_path=args.out, keep_open=args.keep_open,
                 headless=args.headless, timeout_s=args.timeout,
                 use_cookie_fix=not args.no_cookie_fix)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
