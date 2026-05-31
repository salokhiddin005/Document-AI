"""
Google Drive Uploader - save documents directly to Google Drive.
Requires GOOGLE_DRIVE_CREDENTIALS in .env (JSON service account key).
Setup: console.cloud.google.com -> Enable Drive API -> Create Service Account
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from loguru import logger


def is_drive_configured() -> bool:
    """Check if Google Drive credentials are set."""
    creds = os.getenv("GOOGLE_DRIVE_CREDENTIALS", "")
    folder = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
    return bool(creds and folder)


def upload_text_to_drive(
    text: str,
    filename: str,
    doc_id: str,
) -> str:
    """
    Upload text as a .txt file to Google Drive.
    Returns the shareable link.
    """
    creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS", "")
    folder_id  = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "root")

    if not creds_json:
        raise RuntimeError(
            "Google Drive not configured.\n"
            "Add GOOGLE_DRIVE_CREDENTIALS to your Railway Variables.\n"
            "See: console.cloud.google.com"
        )

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaInMemoryUpload
    except ImportError:
        raise RuntimeError(
            "Google Drive libraries not installed.\n"
            "Add to requirements_cloud.txt:\n"
            "  google-api-python-client\n"
            "  google-auth"
        )

    creds_data = json.loads(creds_json)
    scopes     = ["https://www.googleapis.com/auth/drive.file"]
    credentials = service_account.Credentials.from_service_account_info(
        creds_data, scopes=scopes
    )
    service = build("drive", "v3", credentials=credentials)

    # Upload as plain text
    content  = text.encode("utf-8")
    media    = MediaInMemoryUpload(content, mimetype="text/plain; charset=utf-8")
    metadata = {
        "name":    filename,
        "parents": [folder_id],
    }

    file = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, webViewLink",
    ).execute()

    link = file.get("webViewLink", "")
    logger.info(f"Uploaded to Drive: {filename} → {link}")
    return link


def get_setup_instructions() -> str:
    """Return step-by-step setup instructions for Google Drive."""
    return (
        "📁  <b>Google Drive Setup</b>\n\n"
        "To enable Google Drive saving:\n\n"
        "<b>Step 1</b> — Go to console.cloud.google.com\n"
        "<b>Step 2</b> — Create a project → Enable Drive API\n"
        "<b>Step 3</b> — Create Service Account → Download JSON key\n"
        "<b>Step 4</b> — Add to Railway Variables:\n"
        "  <code>GOOGLE_DRIVE_CREDENTIALS</code> = (paste the entire JSON)\n"
        "  <code>GOOGLE_DRIVE_FOLDER_ID</code> = (your Drive folder ID)\n\n"
        "<b>Step 5</b> — Share your Drive folder with the service account email\n\n"
        "💡 The folder ID is in the URL when you open the folder:\n"
        "drive.google.com/drive/folders/<code>THIS_PART</code>"
    )
