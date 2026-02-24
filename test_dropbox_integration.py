"""Test Dropbox Connection for DS_Project

Quick test to verify Dropbox credentials for downloading large model files.
Dropbox is used to store and download models/resources (like anthrokit), not session data.
"""

import os
from datetime import datetime
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Using environment variables only.")


def test_dropbox_credentials():
    """Check if Dropbox credentials are configured."""
    print("\n" + "=" * 70)
    print("DROPBOX CREDENTIALS CHECK")
    print("=" * 70 + "\n")
    
    app_key = os.getenv('DROPBOX_APP_KEY')
    app_secret = os.getenv('DROPBOX_APP_SECRET')
    refresh_token = os.getenv('DROPBOX_REFRESH_TOKEN')
    
    if not app_key:
        print("❌ DROPBOX_APP_KEY not set in environment")
        print("   Set it in .env file: DROPBOX_APP_KEY=your_app_key")
        return False
    
    if not app_secret:
        print("❌ DROPBOX_APP_SECRET not set in environment")
        print("   Set it in .env file: DROPBOX_APP_SECRET=your_app_secret")
        return False
    
    if not refresh_token:
        print("❌ DROPBOX_REFRESH_TOKEN not set in environment")
        print("   Set it in .env file: DROPBOX_REFRESH_TOKEN=your_refresh_token")
        return False
    
    print(f"✅ DROPBOX_APP_KEY: {app_key[:10]}...")
    print(f"✅ DROPBOX_APP_SECRET: {app_secret[:10]}...")
    print(f"✅ DROPBOX_REFRESH_TOKEN: {refresh_token[:20]}...")
    
    return True


def test_dropbox_connection():
    """Test connection to Dropbox."""
    print("\n" + "=" * 70)
    print("DROPBOX CONNECTION TEST")
    print("=" * 70 + "\n")
    
    from src.utils.dropbox_saver import test_dropbox_connection as test_connection
    
    success, message = test_connection()
    print(message)
    
    return success


def test_download_model():
    """Test downloading a model file from Dropbox."""
    print("\n" + "=" * 70)
    print("DROPBOX MODEL DOWNLOAD TEST")
    print("=" * 70 + "\n")
    
    from src.utils.dropbox_saver import download_model_from_dropbox
    
    print("Testing model download functionality...")
    print("(Will skip if model already exists locally)\n")
    
    # Try to download banking77 model
    model_path = download_model_from_dropbox('banking77_logistic_model.pkl')
    
    if model_path and os.path.exists(model_path):
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n✅ Model available at: {model_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
        return True
    else:
        print("\n⚠️  Model not downloaded (check Dropbox configuration or network)")
        return False


def run_all_tests():
    """Run all Dropbox tests."""
    print("\n" + "=" * 70)
    print("DS_PROJECT DROPBOX INTEGRATION TEST")
    print("=" * 70)
    print("\nDropbox Purpose: Download large model files and resources")
    print("Similar to anthrokit pattern - models stored in cloud, downloaded on-demand\n")
    
    # Test 1: Check credentials
    if not test_dropbox_credentials():
        print_setup_instructions()
        return False
    
    # Test 2: Test connection
    if not test_dropbox_connection():
        print("\n⚠️ Connection test failed. Check your credentials.")
        return False
    
    # Test 3: Test model download
    print("\n" + "-" * 70)
    if not test_download_model():
        print("\n⚠️ Download test failed or skipped")
    
    print("\n" + "=" * 70)
    print("✅ DROPBOX INTEGRATION TESTS COMPLETED")
    print("=" * 70 + "\n")
    
    print("Next steps:")
    print("1. Models will be automatically downloaded from Dropbox when needed")
    print("2. Run Streamlit app: streamlit run src/streamlit_app/simple_banking_assistant.py")
    print("3. App will use local models or download from Dropbox if missing")
    
    return True


def print_setup_instructions():
    """Print setup instructions if configuration is missing."""
    print("\n" + "=" * 70)
    print("SETUP INSTRUCTIONS")
    print("=" * 70)
    
    print("""
PURPOSE: Dropbox stores large model files (like .pkl models) for download

1. Use existing Dropbox app credentials (from hicxai-research / anthrokit):
   → Ask project lead for app credentials
   → IMPORTANT: Don't create new app - use existing one!

2. If you absolutely need to create a NEW Dropbox app:
   → Go to: https://www.dropbox.com/developers/apps
   → Click "Create app"
   → Choose: Scoped access, Full Dropbox access
   → Name it: DS-Project-Models
   → Go to Settings tab:
      - Copy App key
      - Copy App secret
   → Go to Permissions tab:
      - Check "files.content.read"  (for downloading)
      - Check "files.content.write" (for uploading models initially)
      - Click "Submit"
   → Generate refresh token:
      https://dropbox.tech/developers/generate-access-tokens-for-your-own-account

3. Add to your .env file:
   DROPBOX_APP_KEY="your_app_key"
   DROPBOX_APP_SECRET="your_app_secret"
   DROPBOX_REFRESH_TOKEN="your_refresh_token"

4. Add to .streamlit/secrets.toml (for Streamlit Cloud deployment):
   DROPBOX_APP_KEY = "your_app_key"
   DROPBOX_APP_SECRET = "your_app_secret"
   DROPBOX_REFRESH_TOKEN = "your_refresh_token"

5. Test the connection:
   python test_dropbox_integration.py

NOTE: Session data goes to GitHub, NOT Dropbox. Dropbox is only for large files.
""")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
