"""Test Dropbox Connection for DS_Project

Thin entry-point that delegates to the reusable
``src.utils.dropbox_integration`` module.  Run directly to verify
Dropbox credentials and connectivity:

    python test_dropbox_integration.py

For programmatic use in notebooks or scripts, import from the module
instead::

    from src.utils.dropbox_integration import (
        check_credentials,
        verify_connection,
        ensure_model_available,
        upload_model,
        list_available_models,
    )
"""

import sys
from pathlib import Path

# Make sure the project root is on the path when run from the repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dropbox_integration import (  # noqa: E402
    check_credentials,
    verify_connection,
    ensure_model_available,
    upload_model,
    list_available_models,
    run_all_checks,
    print_setup_instructions,
)

# ── Re-export individual helpers for backwards-compatible imports ────────────

def test_dropbox_credentials():
    """Check if Dropbox credentials are configured (delegates to check_credentials)."""
    ok, msg = check_credentials()
    print(msg)
    return ok


def test_dropbox_connection():
    """Test live Dropbox connection (delegates to verify_connection)."""
    ok, msg = verify_connection()
    print(msg)
    return ok


def test_download_model(dataset: str = "banking77"):
    """Ensure *dataset* model is available locally, downloading if needed."""
    path = ensure_model_available(dataset)
    return path is not None


def run_all_tests():
    """Run all Dropbox checks (delegates to run_all_checks)."""
    return run_all_checks(download_test=True)



if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as exc:
        print(f"\n❌ Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
