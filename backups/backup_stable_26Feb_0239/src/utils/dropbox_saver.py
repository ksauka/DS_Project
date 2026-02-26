"""Dropbox File Loader for DS_Project

Download large model files and resources from Dropbox that are needed by the agent.
Models stored in Dropbox, downloaded on-demand and cached locally.
"""

import hashlib
import os
import pickle
from typing import Optional, Tuple, List
from pathlib import Path

# Load .env so credentials are available via os.getenv() in all contexts
# (Jupyter notebooks, scripts, etc. — safe no-op if dotenv is not installed)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
except ImportError:
    pass

# Try to import Dropbox SDK
try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
    from dropbox.files import WriteMode, FileMetadata
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    print("⚠️ Dropbox SDK not installed. Install with: pip install dropbox")

# Try to import streamlit for caching (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# Dropbox folder where models are stored
DROPBOX_MODEL_FOLDER = "/ds_project_models"


def _compute_dropbox_content_hash(file_content: bytes) -> str:
    """Compute the Dropbox content hash for in-memory bytes.

    Algorithm: split into 4 MB blocks → SHA-256 each block →
    concatenate digests → SHA-256 the concatenation.
    """
    BLOCK = 4 * 1024 * 1024  # 4 MB
    block_hashes = b"".join(
        hashlib.sha256(file_content[i : i + BLOCK]).digest()
        for i in range(0, max(len(file_content), 1), BLOCK)
    )
    return hashlib.sha256(block_hashes).hexdigest()


def _get_dropbox_client():
    """Get authenticated Dropbox client.
    
    Reads DROPBOX credentials from environment or Streamlit secrets.
    Supports both refresh tokens (recommended) and access tokens.
    
    Returns:
        Dropbox client instance
        
    Raises:
        ValueError if no credentials found
        AuthError if authentication fails
    """
    if not DROPBOX_AVAILABLE:
        raise ImportError("Dropbox SDK not installed. Install with: pip install dropbox")
    
    # Try refresh token first (long-lived, recommended)
    app_key = os.getenv("DROPBOX_APP_KEY")
    app_secret = os.getenv("DROPBOX_APP_SECRET")
    refresh_token = os.getenv("DROPBOX_REFRESH_TOKEN")
    
    # Try Streamlit secrets
    if not all([app_key, app_secret, refresh_token]) and STREAMLIT_AVAILABLE:
        try:
            app_key = app_key or st.secrets.get("DROPBOX_APP_KEY")
            app_secret = app_secret or st.secrets.get("DROPBOX_APP_SECRET")
            refresh_token = refresh_token or st.secrets.get("DROPBOX_REFRESH_TOKEN")
        except Exception:
            pass
    
    # If we have refresh token credentials, use them (recommended)
    if app_key and app_secret and refresh_token:
        try:
            dbx = dropbox.Dropbox(
                app_key=app_key,
                app_secret=app_secret,
                oauth2_refresh_token=refresh_token
            )
            # Test authentication
            dbx.users_get_current_account()
            return dbx
        except AuthError as e:
            raise ValueError(f"Invalid Dropbox refresh token: {e}")
    
    # Fallback to access token (short-lived, expires after 4 hours)
    access_token = os.getenv("DROPBOX_ACCESS_TOKEN")
    if not access_token and STREAMLIT_AVAILABLE:
        try:
            access_token = st.secrets.get("DROPBOX_ACCESS_TOKEN")
        except Exception:
            pass
    
    if not access_token:
        raise ValueError(
            "DROPBOX credentials not found. Set in .env or .streamlit/secrets.toml:\n"
            "Recommended (never expires):\n"
            "  DROPBOX_APP_KEY = 'your_app_key'\n"
            "  DROPBOX_APP_SECRET = 'your_app_secret'\n"
            "  DROPBOX_REFRESH_TOKEN = 'your_refresh_token'\n\n"
            "Or use short-lived access token (expires after 4 hours):\n"
            "  DROPBOX_ACCESS_TOKEN = 'your_token'\n\n"
            "Get credentials at: https://www.dropbox.com/developers/apps"
        )
    
    try:
        dbx = dropbox.Dropbox(access_token)
        # Test authentication
        dbx.users_get_current_account()
        return dbx
    except AuthError as e:
        raise ValueError(f"Invalid Dropbox access token: {e}")


def download_from_dropbox(
    dropbox_path: str,
    local_path: str,
    force_download: bool = False
) -> bool:
    """Download a file from Dropbox to local filesystem.
    
    Args:
        dropbox_path: Path in Dropbox (e.g., '/ds_project_models/banking77_model.pkl')
        local_path: Local destination path (e.g., 'experiments/banking77/model.pkl')
        force_download: Whether to re-download if file exists locally
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> download_from_dropbox(
        ...     dropbox_path='/ds_project_models/banking77_logistic_model.pkl',
        ...     local_path='experiments/banking77/model.pkl'
        ... )
    """
    # Check if file exists and is valid
    if os.path.exists(local_path) and not force_download:
        file_size = os.path.getsize(local_path)
        if file_size > 1000:  # At least 1KB
            print(f"✅ Model found locally: {local_path} ({file_size / 1024 / 1024:.1f} MB)")
            return True
        else:
            print(f"⚠️ Local file corrupted or empty ({file_size} bytes), re-downloading...")
            os.remove(local_path)
    
    # Download from Dropbox
    print(f"⬇️ Downloading from Dropbox: {dropbox_path}")
    
    try:
        dbx = _get_dropbox_client()
        
        # Ensure local directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        metadata, response = dbx.files_download(dropbox_path)
        
        # Save to local file
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        file_size_mb = metadata.size / 1024 / 1024
        print(f"✅ Downloaded successfully: {local_path} ({file_size_mb:.1f} MB)")
        return True
        
    except ApiError as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        print(f"❌ Failed to download from Dropbox: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def upload_model_to_dropbox(
    local_path: str,
    model_name: str,
    overwrite: bool = False
) -> Optional[str]:
    """Upload a model file to Dropbox.
    
    Args:
        local_path: Path to local model file
        model_name: Name to store in Dropbox (e.g., 'banking77_logistic_model.pkl')
        overwrite: Whether to overwrite if file exists in Dropbox
    
    Returns:
        Dropbox path if successful, None otherwise
    
    Example:
        >>> upload_model_to_dropbox(
        ...     local_path='experiments/banking77/model.pkl',
        ...     model_name='banking77_logistic_model.pkl'
        ... )
    """
    if not os.path.exists(local_path):
        print(f"❌ Local file not found: {local_path}")
        return None
    
    dropbox_path = f"{DROPBOX_MODEL_FOLDER}/{model_name}"

    try:
        dbx = _get_dropbox_client()

        # Read file content
        with open(local_path, 'rb') as f:
            file_content = f.read()

        # ── Skip upload if Dropbox already has the same content ──────────────
        try:
            existing = dbx.files_get_metadata(dropbox_path)
            if hasattr(existing, 'content_hash'):
                local_hash = _compute_dropbox_content_hash(file_content)
                if local_hash == existing.content_hash:
                    file_size_mb = len(file_content) / 1024 / 1024
                    print(f"✅ Dropbox already up-to-date: {dropbox_path} ({file_size_mb:.1f} MB) — skipped")
                    return dropbox_path
        except Exception:
            pass  # File doesn't exist yet or metadata unavailable — proceed with upload

        # Upload to Dropbox
        mode = WriteMode.overwrite if overwrite else WriteMode.add

        print(f"⬆️ Uploading to Dropbox: {dropbox_path}")
        metadata = dbx.files_upload(
            file_content,
            dropbox_path,
            mode=mode
        )

        file_size_mb = metadata.size / 1024 / 1024
        print(f"✅ Uploaded successfully: {dropbox_path} ({file_size_mb:.1f} MB)")
        return dropbox_path

    except ApiError as e:
        print(f"❌ Failed to upload to Dropbox: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def list_dropbox_files(folder_path: str = DROPBOX_MODEL_FOLDER) -> List[str]:
    """List files in a Dropbox folder.
    
    Args:
        folder_path: Dropbox folder path (default: /ds_project_models)
    
    Returns:
        List of file names in the folder
    
    Example:
        >>> files = list_dropbox_files()
        >>> print(files)
        ['banking77_logistic_model.pkl', 'clinc150_logistic_model.pkl']
    """
    try:
        dbx = _get_dropbox_client()
        
        result = dbx.files_list_folder(folder_path)
        files = [entry.name for entry in result.entries if isinstance(entry, FileMetadata)]
        
        print(f"📁 Found {len(files)} files in {folder_path}")
        return files
        
    except ApiError as e:
        print(f"❌ Failed to list Dropbox files: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []


# Conditionally add caching decorator if streamlit is available
if STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def download_model_from_dropbox(model_name: str, local_dir: str = "experiments") -> Optional[str]:
        """Download model from Dropbox with caching (Streamlit apps only).
        
        Args:
            model_name: Model filename (e.g., 'banking77_logistic_model.pkl')
            local_dir: Local directory to save to (default: experiments)
        
        Returns:
            Local path to model if successful, None otherwise
        
        Example:
            >>> model_path = download_model_from_dropbox('banking77_logistic_model.pkl')
            >>> with open(model_path, 'rb') as f:
            ...     classifier = pickle.load(f)
        """
        return _download_model_impl(model_name, local_dir)
else:
    def download_model_from_dropbox(model_name: str, local_dir: str = "experiments") -> Optional[str]:
        """Download model from Dropbox (no caching).
        
        Args:
            model_name: Model filename (e.g., 'banking77_logistic_model.pkl')
            local_dir: Local directory to save to (default: experiments)
        
        Returns:
            Local path to model if successful, None otherwise
        
        Example:
            >>> model_path = download_model_from_dropbox('banking77_logistic_model.pkl')
            >>> with open(model_path, 'rb') as f:
            ...     classifier = pickle.load(f)
        """
        return _download_model_impl(model_name, local_dir)


def _download_model_impl(model_name: str, local_dir: str = "experiments") -> Optional[str]:
    """Implementation of model download logic.
    
    Args:
        model_name: Model filename (e.g., 'banking77_logistic_model.pkl')
        local_dir: Local directory to save to
    
    Returns:
        Local path to model if successful, None otherwise
    """
    # Extract dataset name from model filename
    dataset = model_name.split('_')[0]
    local_path = f"{local_dir}/{dataset}/{model_name}"
    
    # Check if already downloaded and valid
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        if file_size > 1000:  # At least 1KB
            print(f"✅ Model found locally: {local_path} ({file_size / 1024 / 1024:.1f} MB)")
            return local_path
        else:
            print(f"⚠️ Local file corrupted ({file_size} bytes), re-downloading...")
            os.remove(local_path)
    
    # Download from Dropbox
    dropbox_path = f"{DROPBOX_MODEL_FOLDER}/{model_name}"
    print(f"📥 Model not found locally. Downloading from Dropbox...")
    
    success = download_from_dropbox(
        dropbox_path=dropbox_path,
        local_path=local_path,
        force_download=False
    )
    
    return local_path if success else None


def load_model(model_name: str, local_dir: str = "experiments"):
    """High-level API: Download model from Dropbox if missing, then load with pickle.
    
    Args:
        model_name: Model filename (e.g., 'banking77_logistic_model.pkl')
        local_dir: Local directory to save to
    
    Returns:
        Loaded model object
    
    Raises:
        FileNotFoundError: If model cannot be downloaded or loaded
    
    Example:
        >>> classifier = load_model('banking77_logistic_model.pkl')
        >>> prediction = classifier.predict(['I want to transfer money'])
    """
    import pickle
    
    # Download if missing
    model_path = download_model_from_dropbox(model_name, local_dir)
    
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Failed to download model: {model_name}\n"
            f"Check Dropbox credentials and network connection"
        )
    
    # Load with pickle
    print(f"📦 Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded successfully")
    return model


def save_model(model, model_name: str, local_dir: str = "experiments", upload_to_dropbox: bool = False):
    """High-level API: Save model locally, optionally upload to Dropbox.
    
    Args:
        model: Model object to save
        model_name: Model filename (e.g., 'banking77_logistic_model.pkl')
        local_dir: Local directory to save to
        upload_to_dropbox: Whether to upload to Dropbox after saving
    
    Returns:
        Tuple of (local_path, dropbox_path) if upload_to_dropbox=True, else just local_path
    
    Example:
        >>> save_model(classifier, 'banking77_logistic_model.pkl', upload_to_dropbox=True)
    """
    import pickle
    
    # Extract dataset name and create local path
    dataset = model_name.split('_')[0]
    local_path = f"{local_dir}/{dataset}/{model_name}"
    
    # Ensure directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save locally
    print(f"💾 Saving model: {local_path}")
    with open(local_path, 'wb') as f:
        pickle.dump(model, f)
    
    file_size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"✅ Model saved locally ({file_size_mb:.1f} MB)")
    
    # Optionally upload to Dropbox
    dropbox_path = None
    if upload_to_dropbox:
        dropbox_path = upload_model_to_dropbox(local_path, model_name, overwrite=True)
    
    return (local_path, dropbox_path) if upload_to_dropbox else local_path


def test_dropbox_connection() -> Tuple[bool, str]:
    """Test Dropbox connection and credentials.
    
    Returns:
        Tuple of (success: bool, message: str)
    
    Example:
        >>> success, message = test_dropbox_connection()
        >>> print(message)
    """
    try:
        dbx = _get_dropbox_client()
        account = dbx.users_get_current_account()
        account_name = account.name.display_name
        return True, f"✅ Connected to Dropbox account: {account_name}"
        
    except Exception as e:
        return False, f"❌ Connection failed: {e}"
