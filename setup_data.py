import shutil
import urllib.request
import zipfile
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR

KG_DIR = ROOT_DIR / "src" / "dataset_raw_KG"
METAQA_URL = "https://github.com/yuyuz/MetaQA/archive/refs/heads/master.zip"
ZIP_PATH = ROOT_DIR / "metaqa_temp.zip"
EXTRACT_DIR = ROOT_DIR / "metaqa_extracted"


def setup_metaqa():
    KG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Downloading MetaQA archive...")
    urllib.request.urlretrieve(METAQA_URL, ZIP_PATH)
    
    print("2. Extracting archive...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
        
    extracted_root = next(EXTRACT_DIR.glob("MetaQA-*"))

    kb_file = extracted_root / "kb.txt"
    if kb_file.exists():
        shutil.copy(kb_file, KG_DIR / "kb.txt")
        print("   -> Copied kb.txt")
        
    for hop in [1, 2, 3]:
        hop_src = extracted_root / f"{hop}-hop"
        hop_dst = KG_DIR / f"{hop}-hop"
        if hop_src.exists():
            if hop_dst.exists():
                shutil.rmtree(hop_dst)
            shutil.copytree(hop_src, hop_dst)
            print(f"   -> Copied {hop}-hop splits")
            
    print("3. Cleaning up temporary files...")
    ZIP_PATH.unlink(missing_ok=True)
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    
    print("\nSetup completed successfully. Everything is ready in src/dataset_raw_KG.")


if __name__ == '__main__':
    setup_metaqa()