import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import gdown

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "datasets"
DATASET_DIR = CACHE_DIR / "dataset_celeb_br"
ENV_PATH = PROJECT_ROOT / ".env"


def load_env(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


env = load_env(ENV_PATH)
GOOGLE_DRIVE_ZIP_URL = env.get("GOOGLE_DRIVE_ZIP_URL")


def get_drive_file_id(url: str) -> str | None:
    if "drive.google.com/drive/folders" in url or "drive.google.com/drive/u" in url:
        return None

    if "drive.google.com/file/d/" in url:
        parts = url.split("/file/d/", 1)[1].split("/")
        return parts[0]

    if "drive.google.com/open" in url and "id=" in url:
        return url.split("id=", 1)[1].split("&")[0]

    if "drive.google.com/uc" in url and "id=" in url:
        return url.split("id=", 1)[1].split("&")[0]

    return None


def normalize_drive_url(url: str) -> str:
    if not url:
        raise RuntimeError(
            "GOOGLE_DRIVE_ZIP_URL não está definido. Defina a URL de download do arquivo ZIP no arquivo .env."
        )

    if "drive.google.com/drive/folders" in url or "drive.google.com/drive/u" in url:
        raise RuntimeError(
            "O link atual é de uma pasta do Google Drive. Use um link direto para um arquivo ZIP público "
            "(por exemplo, https://drive.google.com/uc?id=<FILE_ID>) para baixar tudo de uma vez."
        )

    file_id = get_drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/uc?id={file_id}"

    return url


def extract_zip(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(destination)


def extract_rar(path: Path, destination: Path) -> None:
    try:
        import rarfile
    except ImportError:
        rarfile = None

    if rarfile is not None:
        with rarfile.RarFile(path) as rar_ref:
            rar_ref.extractall(destination)
        return

    unrar_cmd = shutil.which("unrar")
    if unrar_cmd:
        subprocess.run([unrar_cmd, "x", "-y", str(path), str(destination)], check=True)
        return

    sevenz_cmd = shutil.which("7z") or shutil.which("7za")
    if sevenz_cmd:
        subprocess.run([sevenz_cmd, "x", f"-o{destination}", str(path)], check=True)
        return

    raise RuntimeError(
        "O arquivo baixado parece ser um RAR, mas nenhum extrator compatível foi encontrado. "
        "Instale a biblioteca Python rarfile ou um utilitário de linha de comando como unrar ou 7z."
    )


def download_and_extract():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    download_url = normalize_drive_url(GOOGLE_DRIVE_ZIP_URL)
    cwd = os.getcwd()
    os.chdir(CACHE_DIR)
    try:
        print("Baixando dataset do Google Drive com o nome original do arquivo...")
        downloaded = gdown.download(download_url, None, quiet=False)
    finally:
        os.chdir(cwd)

    if not downloaded:
        raise RuntimeError("Falha ao baixar o arquivo do Google Drive.")

    archive_path = Path(downloaded)
    if not archive_path.is_absolute():
        archive_path = CACHE_DIR / archive_path

    print(f"Extraindo arquivos para: {DATASET_DIR}...")

    if zipfile.is_zipfile(archive_path):
        extract_zip(archive_path, DATASET_DIR)
        print("ZIP extraído com sucesso.")
        return

    if archive_path.suffix.lower() == ".rar":
        extract_rar(archive_path, DATASET_DIR)
        print("RAR extraído com sucesso.")
        return

    try:
        extract_zip(archive_path, DATASET_DIR)
        print("ZIP extraído com sucesso.")
        return
    except zipfile.BadZipFile:
        extract_rar(archive_path, DATASET_DIR)
        print("RAR extraído com sucesso.")
        return

if __name__ == "__main__":
    download_and_extract()
