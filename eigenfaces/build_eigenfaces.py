import json
import os
from collections import Counter
from pathlib import Path

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np

DATASET_TAG = "bhaveshmittal/celebrity-face-recognition-dataset"
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "datasets"
DATASET_DIR = CACHE_DIR / "datasets" / "bhaveshmittal" / "celebrity-face-recognition-dataset" / "versions" / "1"
OUTPUT_DIR = PROJECT_ROOT / "results"


# usar o caminho do terminal local + eigenfaces como cache/download
os.environ["KAGGLEHUB_CACHE"] = str(CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not DATASET_DIR.exists() or not any(DATASET_DIR.iterdir()):
    print("Baixando dataset do Kaggle...")
    download_path = kagglehub.dataset_download(DATASET_TAG)
    if not download_path:
        raise RuntimeError("Falha ao baixar dataset com kagglehub")
    dataset_dir = Path(download_path).resolve()
    print(f"Download concluído. Arquivos disponíveis em: {dataset_dir}")
else:
    dataset_dir = DATASET_DIR
    print(f"Dataset encontrado em: {dataset_dir}")

image_entries = []
for celebrity_dir in sorted(dataset_dir.iterdir()):
    if not celebrity_dir.is_dir():
        continue
    for file_path in sorted(celebrity_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".jpg":
            image_entries.append((celebrity_dir.name, file_path))
if not image_entries:
    raise RuntimeError("Nenhuma imagem encontrada em subpastas de celebridades dentro de dataset")


grayscale_images = []
resolutions = Counter()

print(f"Processando {len(image_entries)} imagens...")
for celebrity, image_path in image_entries:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Aviso: não foi possível ler a imagem, ignorando: {image_path}")
        continue

    # mesmo filtro usado pelo prof. Thiago em src/face_recognition.py
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = image_gray.shape
    resolutions[(width, height)] += 1
    grayscale_images.append((celebrity, image_path, image_gray))


if not grayscale_images:
    raise RuntimeError("Nenhuma imagem válida foi lida após aplicar filtro grayscale")


most_common_resolution = resolutions.most_common(1)[0][0]
print(f"Resolução mais comum selecionada: {most_common_resolution[0]}x{most_common_resolution[1]}")

filtered_images = []
skipped_count = 0

for celebrity, image_path, image_gray in grayscale_images:
    height, width = image_gray.shape
    if (width, height) == most_common_resolution:
        filtered_images.append((celebrity, image_path, image_gray))
    else:
        skipped_count += 1

if not filtered_images:
    raise RuntimeError("Nenhuma imagem sobrou após filtrar pela resolução mais comum")

print(f"Imagens usadas: {len(filtered_images)}")
print(f"Imagens puladas por resolução diferente: {skipped_count}")


flattened_vectors = []
label_map = {}

for column_index, (celebrity, image_path, image_gray) in enumerate(filtered_images):
    flattened_vectors.append(image_gray.flatten())
    relative_path = os.path.relpath(image_path, dataset_dir)
    label_map[relative_path] = {
        "column": column_index,
        "path": str(image_path),
        "celebrity": celebrity,
    }


X = np.column_stack(flattened_vectors)

# calcular face média (X_bar) subtraindo por linha (pixel) - para cada linha, calcula media dos valores daquela linha (pixel) ao longo de todas as colunas (imagens) e subtrai essa média de cada valor naquela linha. O resultado é uma matriz onde cada pixel tem média zero, o que é importante para o cálculo do SVD posteriormente.
X_bar = X.mean(axis=1)

# centralizar os dados
# X_bar.reshape(-1, 1): transforma X_bar de um vetor para uma matriz coluna de X_bar x X_bar x .. x X_bar, n vezes 
X_centered = X - X_bar.reshape(-1, 1)

# calcular SVD
print("Calculando SVD...")
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False) # full_matrices=False é a SVD "magra"

projections = U.T @ X_centered

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT_DIR / "X.npy", X)
np.save(OUTPUT_DIR / "X_bar.npy", X_bar)
np.save(OUTPUT_DIR / "X_centered.npy", X_centered)
np.save(OUTPUT_DIR / "U.npy", U)
np.save(OUTPUT_DIR / "S.npy", S)
np.save(OUTPUT_DIR / "Vt.npy", Vt)
np.save(OUTPUT_DIR / "projections.npy", projections)

metadata = {
    "image_shape": [filtered_images[0][2].shape[0], filtered_images[0][2].shape[1]],
    "resolution": [most_common_resolution[0], most_common_resolution[1]],
    "sample_count": int(X.shape[1]),
    "skipped_count": int(skipped_count),
}

with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as json_file:
    json.dump(label_map, json_file, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as json_file:
    json.dump(metadata, json_file, ensure_ascii=False, indent=2)

mean_face_img = X_bar.reshape((filtered_images[0][2].shape[0], filtered_images[0][2].shape[1]))
mean_face_path = OUTPUT_DIR / "mean_face.png"
plt.figure(figsize=(6, 6))
plt.imshow(mean_face_img, cmap="gray")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(mean_face_path, bbox_inches="tight", pad_inches=0)
plt.close()

print(f"Matriz X salva em: {OUTPUT_DIR / 'X.npy'}")
print(f"Shape de X: {X.shape}")
print(f"Face média X_bar salva em: {OUTPUT_DIR / 'X_bar.npy'}")
print(f"Shape de X_bar: {X_bar.shape}")
print(f"X centralizado salvo em: {OUTPUT_DIR / 'X_centered.npy'}")
print(f"Shape de X_centered: {X_centered.shape}")
print(f"Eigenfaces U salvo em: {OUTPUT_DIR / 'U.npy'}")
print(f"Shape de U: {U.shape}")
print(f"Valores singulares S salvo em: {OUTPUT_DIR / 'S.npy'}")
print(f"Shape de S: {S.shape}")
print(f"Vt salvo em: {OUTPUT_DIR / 'Vt.npy'}")
print(f"Shape de Vt: {Vt.shape}")
print(f"Projeções salvas em: {OUTPUT_DIR / 'projections.npy'}")
print(f"Mapeamento JSON salvo em: {OUTPUT_DIR / 'label_map.json'}")
print(f"Plot da face média salvo em: {mean_face_path}")