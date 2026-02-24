"""GitHub Saver Utility

Save user session data directly to a private GitHub repository using the GitHub API.
Requires a GitHub personal access token with repo permissions.
"""

import requests
import base64
import os
from typing import Optional


def save_to_github(
    repo: str,
    path: str,
    content: str,
    commit_message: str,
    github_token: str
) -> bool:
    """
    Save content to a file in a GitHub repo (creates or updates the file).
    
    Args:
        repo: 'username/repo' (e.g., 'ksauka/hicxai-data-private')
        path: path in the repo (e.g., 'sessions/2026-02-23/session_abc123.json')
        content: string content to save
        commit_message: commit message
        github_token: GitHub personal access token with repo permissions
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> save_to_github(
        ...     repo='ksauka/hicxai-data-private',
        ...     path='sessions/2026-02-23/session_abc123.json',
        ...     content=json.dumps(data, indent=2),
        ...     commit_message='Session data: participant P123',
        ...     github_token=os.getenv('GITHUB_DATA_TOKEN')
        ... )
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if file exists (to get SHA for update)
    try:
        r = requests.get(api_url, headers=headers, timeout=10)
        if r.status_code == 200:
            sha = r.json()['sha']
        else:
            sha = None
    except Exception as e:
        print(f"Warning: Could not check file existence: {e}")
        sha = None
    
    # Prepare content
    data = {
        "message": commit_message,
        "content": base64.b64encode(content.encode('utf-8')).decode('utf-8')
    }
    
    if sha:
        data["sha"] = sha  # Required for updates
    
    # Create or update file
    try:
        response = requests.put(api_url, json=data, headers=headers, timeout=30)
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully saved to GitHub: {path}")
            return True
        else:
            print(f"❌ GitHub API error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print("❌ GitHub API timeout")
        return False
    except Exception as e:
        print(f"❌ GitHub save failed: {e}")
        return False


def test_github_connection(github_token: str, repo: str) -> tuple[bool, str]:
    """
    Test GitHub API connection and repo access.
    
    Args:
        github_token: GitHub personal access token
        repo: 'username/repo'
    
    Returns:
        (success, message) tuple
    """
    try:
        api_url = f"https://api.github.com/repos/{repo}"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            repo_data = response.json()
            return True, f"✅ Connected to {repo_data['full_name']}"
        elif response.status_code == 404:
            return False, f"❌ Repository '{repo}' not found or no access"
        elif response.status_code == 401:
            return False, "❌ Invalid GitHub token"
        else:
            return False, f"❌ GitHub API error: {response.status_code}"
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"
