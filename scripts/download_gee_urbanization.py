"""
Download the econp_gee_urbanization CSV exports from Google Drive.

Authenticates with the earthengine refresh token, which carries the
Drive scope.

Usage: python3 scripts/download_gee_urbanization.py [--out data/gee_exports]
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.oauth2.credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import ee.oauth as eo

CRED_PATH = Path.home() / ".config" / "earthengine" / "credentials"
FOLDER = "econp_gee_urbanization"


def load_credentials():
    data = json.loads(CRED_PATH.read_text())
    creds = google.oauth2.credentials.Credentials(
        token=None,
        refresh_token=data["refresh_token"],
        client_id=eo.CLIENT_ID,
        client_secret=eo.CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=data.get("scopes"),
    )
    creds.refresh(Request())
    return creds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gee_exports")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    svc = build("drive", "v3", credentials=load_credentials(),
                cache_discovery=False)
    res = svc.files().list(
        q=f"name = '{FOLDER}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        spaces="drive", fields="files(id,name)").execute()
    items = res.get("files", [])
    if not items:
        sys.exit(f"folder {FOLDER} not found on Drive")
    fid = items[0]["id"]

    files, token = [], None
    while True:
        r = svc.files().list(
            q=f"'{fid}' in parents and name contains '.csv' and trashed = false",
            spaces="drive", fields="nextPageToken, files(id,name,size)",
            pageToken=token).execute()
        files += r.get("files", [])
        token = r.get("nextPageToken")
        if not token:
            break

    print(f"{len(files)} CSVs in Drive/{FOLDER}")
    for f in sorted(files, key=lambda x: x["name"]):
        dest = Path(args.out) / f["name"]
        if dest.exists() and dest.stat().st_size == int(f.get("size", -1)):
            print(f"  skip (cached) {f['name']}")
            continue
        req = svc.files().get_media(fileId=f["id"])
        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        dest.write_bytes(buf.getvalue())
        print(f"  downloaded {f['name']} ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
