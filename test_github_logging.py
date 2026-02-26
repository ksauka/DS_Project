#!/usr/bin/env python
"""Test GitHub logging integration"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.github_saver import test_github_connection, save_to_github
from src.utils.data_logger import DataLogger
import json
from datetime import datetime


def test_connection():
    """Test GitHub connection"""
    print("=" * 70)
    print("TESTING GITHUB CONNECTION")
    print("=" * 70)
    
    repo = os.getenv('GITHUB_DATA_REPO')
    token = os.getenv('GITHUB_DATA_TOKEN')
    
    if not repo:
        print("❌ GITHUB_DATA_REPO not set in environment")
        print("   Set it in .env file: GITHUB_DATA_REPO=ksauka/hicxai-data-private")
        return False
    
    if not token:
        print("❌ GITHUB_DATA_TOKEN not set in environment")
        print("   Generate token at: https://github.com/settings/tokens")
        print("   Required scope: 'repo' (Full control of private repositories)")
        return False
    
    print(f"\nRepository: {repo}")
    print(f"Token: {token[:10]}...{token[-5:] if len(token) > 15 else ''}")
    
    success, message = test_github_connection(token, repo)
    print(f"\n{message}\n")
    
    return success


def test_save_sample_session():
    """Test saving a sample session"""
    print("=" * 70)
    print("TESTING SESSION SAVE")
    print("=" * 70)
    
    repo = os.getenv('GITHUB_DATA_REPO')
    token = os.getenv('GITHUB_DATA_TOKEN')
    
    if not repo or not token:
        print("⚠️ Skipping save test (GitHub not configured)")
        return False
    
    # Create sample logger
    logger = DataLogger(
        participant_id="TEST_USER_001",
        condition="test",
        session_id="test_session_123"
    )
    
    # Add sample query result
    sample_result = {
        'session_id': 'test_session_123',
        'query_index': 0,
        'query_text': 'I want to transfer money',
        'true_intent': 'transfer',
        'predicted_intent': 'transfer',
        'confidence': 0.87,
        'num_clarification_turns': 1,
        'is_correct': True,
        'interaction_time_seconds': 23.5,
        'conversation_transcript': 'User: I want to transfer money\nAssistant: ...',
        'timestamp': datetime.now().isoformat(),
        'feedback_clarity': 5,
        'feedback_confidence': 4,
        'feedback_comment': 'Very clear',
        'feedback_submitted': True
    }
    
    logger.log_query_result(sample_result)
    
    # Set final feedback
    logger.set_final_feedback({
        "overall_rating": 4,
        "trust": 4,
        "ease_of_use": 5,
        "would_recommend": "Probably",
        "additional_comments": "Test session - this is a test"
    })
    
    # Print data structure
    print("\nSample session data:")
    print("-" * 70)
    data = logger.build_final_data()
    print(json.dumps(data, indent=2)[:1000] + "...\n")
    
    # Attempt save to GitHub
    print("Attempting to save to GitHub...")
    success = logger.save_to_github(repo, token)
    
    if success:
        print("\n✅ TEST SESSION SAVED SUCCESSFULLY!")
        print(f"\nCheck your repository: https://github.com/{repo}/tree/main/sessions")
    else:
        print("\n❌ Save failed (data saved locally as fallback)")
        print("   Check: data/sessions/ directory")
    
    return success


def show_instructions():
    """Show setup instructions"""
    print("\n" + "=" * 70)
    print("GITHUB LOGGING SETUP INSTRUCTIONS")
    print("=" * 70)
    
    print("""
1. This project uses the existing PRIVATE GitHub repository:
   → Repository: ksauka/hicxai-data-private
   → URL: https://github.com/ksauka/hicxai-data-private
   → Status: Already created ✓

2. Generate a Personal Access Token (if you don't have one):
   → Go to: https://github.com/settings/tokens
   → Click "Generate new token (classic)"
   → Note: "DS Project Data Logging"
   → Expiration: 90 days (recommended)
   → Scopes: Check '✓ repo' (Full control of private repositories)
   → Click "Generate token"
   → COPY THE TOKEN IMMEDIATELY (you won't see it again!)

3. Add to your .env file:
   GITHUB_DATA_REPO="ksauka/hicxai-data-private"
   GITHUB_DATA_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

4. Test the connection:
   python test_github_logging.py

For full documentation, see: GITHUB_LOGGING_SETUP.md
""")


if __name__ == "__main__":
    print("\n🔧 DS_Project GitHub Logging Test")
    print("=" * 70)
    
    # Test connection
    connection_ok = test_connection()
    
    if not connection_ok:
        show_instructions()
        sys.exit(1)
    
    # Test save
    print()
    save_ok = test_save_sample_session()
    
    if save_ok:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nYour GitHub logging is configured correctly.")
        print("Session data will automatically save to GitHub during user studies.")
        print("\nNext steps:")
        print("1. Deploy Streamlit app to Streamlit Cloud")
        print("2. Add secrets in Streamlit Cloud settings:")
        print("   → GITHUB_DATA_REPO")
        print("   → GITHUB_DATA_TOKEN")
        print("3. Run pilot study to verify end-to-end flow")
    else:
        print("\n" + "=" * 70)
        print("⚠️ SAVE TEST FAILED")
        print("=" * 70)
        print("\nLocal fallback will work, but GitHub sync is not functional.")
        print("Check:")
        print("- Token has 'repo' scope")
        print("- Repository exists and is accessible")
        print("- Network connection is working")
