import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DIR = CACHE_DIR / "dataset_celeb_br_processed"
TRAIN_DIR = PROCESSED_DIR / "train"
OUTPUT_DIR = PROJECT_ROOT / "results"

EPS = 1e-10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP_FILE = OUTPUT_DIR / "label_map_train.json"
if not LABEL_MAP_FILE.exists():
    raise RuntimeError("Arquivo de label map de treino não encontrado. Execute process_dataset.py primeiro.")

with LABEL_MAP_FILE.open("r", encoding="utf-8") as f:
    label_map = json.load(f)

sample_items = sorted(label_map.items(), key=lambda item: item[1]["column"])
if not sample_items:
    raise RuntimeError("Nenhum exemplo de treino encontrado no label_map_train.json")

processed_arrays = []
for relative_path, info in sample_items:
    npy_path = Path(info["npy_path"])
    if not npy_path.exists():
        raise RuntimeError(f"Arquivo de dados .npy não encontrado: {npy_path}")
    processed_arrays.append(np.load(npy_path).astype(np.float32))

vector_size = processed_arrays[0].shape[0]
X = np.column_stack([arr.flatten() for arr in processed_arrays])

X_bar = X.mean(axis=1)
X_centered = X - X_bar.reshape(-1, 1)
print(f"Matriz X: {X.shape}, vetor por amostra: {vector_size}")
print(f"Média X_bar shape: {X_bar.shape}, X_centered shape: {X_centered.shape}")

print("Calculando SVD...")
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
variance = S**2
variance_ratio = variance / variance.sum()
cumulative = np.cumsum(variance_ratio)
k_max = int(np.searchsorted(cumulative, 0.95) + 1)
print(f"95% da variância explicada em {k_max} componentes")
print(f"Top 5 variância explicada: {variance_ratio[:5].tolist()}")

natural_k = U.shape[1]
print(f"Usando k_max={k_max} para 95% de variância, natural k={natural_k}, U shape={U.shape}, S shape={S.shape}, Vt shape={Vt.shape}")

projections = U.T @ X_centered
print(f"Projeções shape: {projections.shape}")

np.save(OUTPUT_DIR / "X.npy", X)
np.save(OUTPUT_DIR / "X_bar.npy", X_bar)
np.save(OUTPUT_DIR / "X_centered.npy", X_centered)
np.save(OUTPUT_DIR / "U.npy", U)
np.save(OUTPUT_DIR / "S.npy", S)
np.save(OUTPUT_DIR / "Vt.npy", Vt)
np.save(OUTPUT_DIR / "projections.npy", projections)

# Lê o metadata original para pegar resolução e atualizar infos
process_metadata_path = OUTPUT_DIR / "metadata.json"
if process_metadata_path.exists():
    with process_metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = {"resolution": [256, 256]}

metadata.update({
    "train_sample_count": X.shape[1],
    "split_ratio": [0.8, 0.2],
    "k_max": k_max,
    "variance_explained_95": float(cumulative[k_max - 1]),
})

with (OUTPUT_DIR / "svd_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

res_y, res_x = metadata["resolution"]
single_img_size = res_y * res_x

mean_face_img = X_bar[:single_img_size].reshape((res_y, res_x))
plt.figure(figsize=(6, 6))
plt.imshow(mean_face_img, cmap="gray")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(OUTPUT_DIR / "mean_face.png", bbox_inches="tight", pad_inches=0)
plt.close()

u_dir = OUTPUT_DIR / "U"
u_dir.mkdir(parents=True, exist_ok=True)
for index in range(min(5, U.shape[1])):
    u_image = U[:single_img_size, index].reshape((res_y, res_x))
    plt.figure(figsize=(6, 6))
    plt.imshow(u_image, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(u_dir / f"U_{index + 1}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

print(f"SVD concluída. Resultados salvos em: {OUTPUT_DIR}")
