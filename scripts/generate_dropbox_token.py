"""Generate a Dropbox OAuth2 refresh token and save it to .env.

Usage:
    python scripts/generate_dropbox_token.py

Reads DROPBOX_APP_KEY and DROPBOX_APP_SECRET from .env, walks you through
the one-time browser authorization, then appends DROPBOX_REFRESH_TOKEN to .env.
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
try:
    import dropbox
    from dropbox import DropboxOAuth2FlowNoRedirect
except ImportError:
    print("❌  Dropbox SDK not installed.  Run:  pip install dropbox")
    sys.exit(1)

APP_KEY    = os.getenv("DROPBOX_APP_KEY", "").strip()
APP_SECRET = os.getenv("DROPBOX_APP_SECRET", "").strip()

if not APP_KEY or not APP_SECRET:
    print("❌  DROPBOX_APP_KEY and DROPBOX_APP_SECRET must be set in .env first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check if token already exists
# ---------------------------------------------------------------------------
existing_token = os.getenv("DROPBOX_REFRESH_TOKEN", "").strip()
if existing_token:
    print(f"✅  DROPBOX_REFRESH_TOKEN already present in .env.")
    print(f"   Token prefix: {existing_token[:6]}…")
    print("   Nothing to do.  Delete the line from .env to regenerate.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# OAuth2 no-redirect flow
# ---------------------------------------------------------------------------
print("=" * 60)
print("DROPBOX REFRESH TOKEN GENERATOR")
print("=" * 60)
print(f"\nApp Key    : {APP_KEY}")
print(f"App Secret : {APP_SECRET[:4]}…")
print()

auth_flow = DropboxOAuth2FlowNoRedirect(
    consumer_key=APP_KEY,
    consumer_secret=APP_SECRET,
    token_access_type="offline",  # offline = long-lived refresh token
)

authorize_url = auth_flow.start()

print("STEP 1: Open this URL in your browser and click 'Allow':")
print()
print(f"  {authorize_url}")
print()
print("STEP 2: Copy the authorization code shown after you approve.")
print()

auth_code = input("STEP 3: Paste the authorization code here and press Enter: ").strip()

if not auth_code:
    print("❌  No code entered.  Aborting.")
    sys.exit(1)

try:
    oauth_result = auth_flow.finish(auth_code)
except Exception as e:
    print(f"❌  Failed to exchange auth code: {e}")
    sys.exit(1)

refresh_token = oauth_result.refresh_token
if not refresh_token:
    print("❌  No refresh token returned.  "
          "Make sure 'token_access_type=offline' is supported by your app.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Write token to .env
# ---------------------------------------------------------------------------
env_path = project_root / ".env"
with env_path.open("a") as fh:
    fh.write(f"\nDROPBOX_REFRESH_TOKEN = \"{refresh_token}\"\n")

print()
print("=" * 60)
print("✅  SUCCESS!")
print("=" * 60)
print(f"   Refresh token written to: {env_path}")
print(f"   Token prefix: {refresh_token[:6]}…")
print()
print("Re-run STEP 0 in the notebook — Dropbox upload will now work.")
