# scripts/fetch_assets.py
import os
import gdown
import zipfile

def download_and_extract(file_id, output_zip, extract_to, force=False):
    if force and os.path.exists(extract_to):
        import shutil
        shutil.rmtree(extract_to)

    if not os.path.exists(extract_to):
        print(f"[INFO] Downloading {output_zip} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_zip, quiet=False)

        print(f"[INFO] Extracting {output_zip}...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"[SUCCESS] Extracted to '{extract_to}'")
    else:
        print(f"[SKIPPED] '{extract_to}' already exists.")
