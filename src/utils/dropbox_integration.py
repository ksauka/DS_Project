"""Dropbox Integration Utilities for DS_Project

Provides a clean, reusable API for all Dropbox operations needed by the
project: credential checking, connectivity verification, on-demand model
download, model uploading, and Dropbox folder listing.

Usage examples
--------------
From a script or notebook::

    from src.utils.dropbox_integration import (
        check_credentials,
        verify_connection,
        ensure_model_available,
        upload_model,
        list_available_models,
    )

    ok, msg = verify_connection()
    model_path = ensure_model_available("banking77", experiments_dir=Path("experiments/banking77"))

From the command line::

    python -m src.utils.dropbox_integration          # run all checks
    python -m src.utils.dropbox_integration --upload experiments/banking77/banking77_logistic_model.pkl
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Try to load .env, but don't fail if dotenv is not installed
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # Use absolute path so this works from any working directory
    # (e.g. notebooks/ inside Jupyter, or project root in scripts)
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def check_credentials() -> Tuple[bool, str]:
    """Check whether all required Dropbox credentials are present in the
    environment (or Streamlit secrets).

    Returns
    -------
    (ok, message) : tuple[bool, str]
        *ok* is ``True`` when all three variables are set; *message* contains
        a human-readable summary.
    """
    keys = {
        "DROPBOX_APP_KEY":      os.getenv("DROPBOX_APP_KEY"),
        "DROPBOX_APP_SECRET":   os.getenv("DROPBOX_APP_SECRET"),
        "DROPBOX_REFRESH_TOKEN": os.getenv("DROPBOX_REFRESH_TOKEN"),
    }

    # Try Streamlit secrets as fallback
    try:
        import streamlit as st
        for k in keys:
            if not keys[k]:
                keys[k] = st.secrets.get(k)
    except Exception:
        pass

    missing = [k for k, v in keys.items() if not v]
    if missing:
        msg = (
            f"❌ Missing Dropbox credentials: {', '.join(missing)}\n"
            "   Set them in your .env file or .streamlit/secrets.toml — see\n"
            "   print_setup_instructions() for details."
        )
        return False, msg

    lines = [f"✅ {k}: {v[:12]}..." for k, v in keys.items() if v]
    return True, "\n".join(lines)


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def verify_connection() -> Tuple[bool, str]:
    """Test the live Dropbox connection using the credentials in the
    environment.

    Returns
    -------
    (ok, message) : tuple[bool, str]
    """
    from src.utils.dropbox_saver import test_dropbox_connection
    return test_dropbox_connection()


# ---------------------------------------------------------------------------
# Model availability (download-on-demand)
# ---------------------------------------------------------------------------

def ensure_model_available(
    dataset_name: str,
    experiments_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    force_download: bool = False,
) -> Optional[Path]:
    """Return the local path to the trained model, downloading it from
    Dropbox if it is not already present.

    Parameters
    ----------
    dataset_name : str
        E.g. ``"banking77"`` or ``"clinc150"``.
    experiments_dir : Path, optional
        Directory that holds the ``.pkl`` file.  Defaults to
        ``{project_root}/experiments/{dataset_name}/``.
    project_root : Path, optional
        Project root.  Inferred from this file's location if not given.
    force_download : bool
        Re-download even when the local file already exists.

    Returns
    -------
    Path or None
        Local path to the model file, or ``None`` if not available.
    """
    if project_root is None:
        # src/utils/dropbox_integration.py  →  ../../  =  project root
        project_root = Path(__file__).parent.parent.parent

    if experiments_dir is None:
        experiments_dir = project_root / "experiments" / dataset_name

    model_filename = f"{dataset_name}_logistic_model.pkl"
    local_path = experiments_dir / model_filename
    dropbox_path = f"/ds_project_models/{model_filename}"

    if local_path.exists() and not force_download:
        size_mb = local_path.stat().st_size / (1024 ** 2)
        print(f"✅ Model already available locally: {local_path}  ({size_mb:.2f} MB)")
        return local_path

    print(f"⬇️  Downloading {model_filename} from Dropbox …")
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from src.utils.dropbox_saver import download_from_dropbox
        ok = download_from_dropbox(dropbox_path, str(local_path), force_download=force_download)
        if ok and local_path.exists():
            size_mb = local_path.stat().st_size / (1024 ** 2)
            print(f"✅ Downloaded to: {local_path}  ({size_mb:.2f} MB)")
            return local_path
        else:
            print(f"❌ Download failed for {dataset_name}")
            return None
    except Exception as exc:
        print(f"❌ Dropbox download error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Model upload
# ---------------------------------------------------------------------------

def upload_model(
    local_path: Path,
    model_name: Optional[str] = None,
    overwrite: bool = False,
) -> bool:
    """Upload a local model file to the Dropbox ``/ds_project_models/`` folder.

    Parameters
    ----------
    local_path : Path
        Path to the ``.pkl`` file to upload.
    model_name : str, optional
        Remote filename; defaults to the basename of *local_path*.
    overwrite : bool
        Overwrite the remote file if it already exists.

    Returns
    -------
    bool
        ``True`` on success.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"❌ File not found: {local_path}")
        return False

    name = model_name or local_path.name

    try:
        from src.utils.dropbox_saver import upload_model_to_dropbox
        dropbox_path = upload_model_to_dropbox(str(local_path), name, overwrite=overwrite)
        if dropbox_path:
            return True
        print(f"❌ Upload returned no path for {name}")
        return False
    except Exception as exc:
        print(f"❌ Upload error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_available_models() -> List[str]:
    """Return the list of filenames in the Dropbox ``/ds_project_models/``
    folder.

    Returns
    -------
    list[str]
        Filenames (not full paths), e.g.
        ``['banking77_logistic_model.pkl', 'clinc150_logistic_model.pkl']``.
    """
    try:
        from src.utils.dropbox_saver import list_dropbox_files
        files = list_dropbox_files()
        return files or []
    except Exception as exc:
        print(f"❌ Could not list Dropbox files: {exc}")
        return []


# ---------------------------------------------------------------------------
# Comprehensive health check (replaces run_all_tests in the old script)
# ---------------------------------------------------------------------------

def run_all_checks(download_test: bool = True) -> bool:
    """Run the full suite of Dropbox health checks.

    1. Credential presence
    2. Live connection
    3. (Optional) download a model to verify end-to-end functionality

    Parameters
    ----------
    download_test : bool
        Whether to attempt a model download as part of the check.

    Returns
    -------
    bool
        ``True`` when all enabled checks pass.
    """
    _sep = "=" * 65

    print(f"\n{_sep}")
    print("DS_PROJECT  DROPBOX INTEGRATION CHECK")
    print(_sep)

    # 1 ── credentials ─────────────────────────────────────────────────────
    print("\n[1/3] Credentials …")
    cred_ok, cred_msg = check_credentials()
    print(cred_msg)
    if not cred_ok:
        print_setup_instructions()
        return False

    # 2 ── connection ──────────────────────────────────────────────────────
    print("\n[2/3] Connection …")
    conn_ok, conn_msg = verify_connection()
    print(conn_msg)
    if not conn_ok:
        print("⚠️  Connection failed — check credentials and network.")
        return False

    # 3 ── available models ────────────────────────────────────────────────
    print("\n[3/3] Models in Dropbox …")
    models = list_available_models()
    if models:
        for m in models:
            print(f"   • {m}")
    else:
        print("   (none found in /ds_project_models/)")

    # optional download smoke-test
    if download_test and models:
        first = models[0]
        dataset = first.replace("_logistic_model.pkl", "").replace("_model.pkl", "")
        print(f"\n   Smoke-test: ensuring '{dataset}' model is available locally …")
        ensure_model_available(dataset)

    print(f"\n{_sep}")
    print("✅ ALL CHECKS PASSED")
    print(_sep)
    print("\nNotes:")
    print("  • Models are downloaded on-demand when the Streamlit app starts")
    print("  • Streamlit app: streamlit run src/streamlit_app/simple_banking_assistant.py")
    print("  • To upload a new model: upload_model(Path('experiments/x/x_logistic_model.pkl'))")
    return True


# ---------------------------------------------------------------------------
# Setup instructions
# ---------------------------------------------------------------------------

def print_setup_instructions() -> None:
    """Print Dropbox credential setup instructions to stdout."""
    print("""
DROPBOX SETUP INSTRUCTIONS
===========================
PURPOSE: Dropbox stores large model .pkl files for on-demand download.

1. Use the existing project Dropbox app credentials.
   → Ask the project lead for DROPBOX_APP_KEY / DROPBOX_APP_SECRET /
     DROPBOX_REFRESH_TOKEN (never create a duplicate app).

2. Add them to your local .env file:
     DROPBOX_APP_KEY="…"
     DROPBOX_APP_SECRET="…"
     DROPBOX_REFRESH_TOKEN="…"

3. For Streamlit Cloud, add to .streamlit/secrets.toml:
     DROPBOX_APP_KEY = "…"
     DROPBOX_APP_SECRET = "…"
     DROPBOX_REFRESH_TOKEN = "…"

4. Verify: python -m src.utils.dropbox_integration

Reference: https://www.dropbox.com/developers/apps
""")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="DS_Project Dropbox integration utility"
    )
    parser.add_argument(
        "--upload", metavar="PATH",
        help="Upload a local .pkl file to Dropbox /ds_project_models/",
    )
    parser.add_argument(
        "--download", metavar="DATASET",
        help="Download model for DATASET (e.g. banking77) to experiments/",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List models available in Dropbox",
    )
    parser.add_argument(
        "--no-download-test", dest="download_test", action="store_false",
        default=True,
        help="Skip the download smoke-test in run_all_checks()",
    )
    args = parser.parse_args()

    if args.upload:
        ok = upload_model(Path(args.upload))
        sys.exit(0 if ok else 1)

    if args.download:
        path = ensure_model_available(args.download)
        sys.exit(0 if path else 1)

    if args.list:
        models = list_available_models()
        if models:
            print("Models in Dropbox /ds_project_models/:")
            for m in models:
                print(f"  • {m}")
        else:
            print("No models found.")
        sys.exit(0)

    # Default: run all checks
    ok = run_all_checks(download_test=args.download_test)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
