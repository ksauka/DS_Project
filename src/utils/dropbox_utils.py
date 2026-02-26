"""Dropbox utility for DS_Project model management.

Workflow
--------
  STEP 0  →  train_model()  →  upload_model()  →  Dropbox /ds_project_models/

Command-line usage
------------------
# Upload a locally trained model to Dropbox
python dropbox_utils.py --upload experiments/banking77/banking77_logistic_model.pkl

"""

import sys
from pathlib import Path

# Make sure project root is on the path when run directly
# src/utils/dropbox_utils.py → ../../  =  project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.dropbox_integration import (   # noqa: E402
    check_credentials,
    verify_connection,
    upload_model,
    list_available_models,
    run_all_checks,
    print_setup_instructions,
)

# Re-export for notebook imports:  from dropbox_utils import upload_model
__all__ = [
    "check_credentials",
    "verify_connection",
    "upload_model",
    "list_available_models",
    "run_all_checks",
    "print_setup_instructions",
]


def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        prog="dropbox_utils",
        description="DS_Project Dropbox model uploader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--upload", metavar="PATH",
                        help="Upload a trained .pkl file to Dropbox /ds_project_models/")
    parser.add_argument("--list", action="store_true",
                        help="List models currently stored in Dropbox /ds_project_models/")
    parser.add_argument("--check", action="store_true",
                        help="Verify Dropbox credentials and connectivity")
    args = parser.parse_args()

    if args.check or not any([args.upload, args.list]):
        ok = run_all_checks(download_test=False)
        sys.exit(0 if ok else 1)

    if args.list:
        models = list_available_models()
        if models:
            print("Models in Dropbox /ds_project_models/:")
            for m in models:
                print(f"  • {m}")
        else:
            print("No models found in Dropbox.")
        sys.exit(0)

    if args.upload:
        ok = upload_model(Path(args.upload), overwrite=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    _cli()
