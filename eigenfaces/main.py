import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT


def main() -> None:
    scripts = [
        "download_dataset.py",
        "process_dataset.py",
        "build_eigenfaces.py",
        "evaluate_validation.py",
    ]

    for script_name in scripts:
        script_path = SCRIPTS_DIR / script_name
        print(f"\n=== Executando {script_name} ===")
        subprocess.run([sys.executable, str(script_path)], check=True)

    print("\nPipeline concluída com sucesso.")


if __name__ == "__main__":
    main()
