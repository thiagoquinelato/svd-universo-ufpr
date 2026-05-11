import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results"


def preprocess_image(image_path, target_shape):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Não foi possível ler a imagem: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_gray.shape != target_shape:
        image_gray = cv2.resize(image_gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

    return image_gray.flatten()


parser = argparse.ArgumentParser(description="Avalia uma imagem usando o modelo Eigenfaces")
parser.add_argument("image", type=Path, help="Caminho da imagem a ser avaliada")
args = parser.parse_args()

with open(OUTPUT_DIR / "metadata.json", "r", encoding="utf-8") as json_file:
    metadata = json.load(json_file)

with open(OUTPUT_DIR / "label_map.json", "r", encoding="utf-8") as json_file:
    label_map = json.load(json_file)

X_bar = np.load(OUTPUT_DIR / "X_bar.npy")
U = np.load(OUTPUT_DIR / "U.npy")
projections = np.load(OUTPUT_DIR / "projections.npy")
image_shape = tuple(metadata["image_shape"])

query = preprocess_image(args.image, image_shape)
query_centered = query - X_bar
query_projection = U.T @ query_centered

sample_rows = sorted(label_map.items(), key=lambda item: item[1]["column"])
distances = np.linalg.norm(projections.T - query_projection, axis=1)

ranked = []
for (relative_path, info), distance in zip(sample_rows, distances):
    ranked.append(
        {
            "celebrity": info["celebrity"],
            "path": info["path"],
            "relative_path": relative_path,
            "column": info["column"],
            "distance": float(distance),
        }
    )

ranked.sort(key=lambda item: item["distance"])
best = ranked[0]

print(f"Predição principal: {best['celebrity']}")
print(f"Imagem de referência mais próxima: {best['path']}")
print(f"Distância: {best['distance']:.4f}")
print("Top 3:")
for item in ranked[:3]:
    print(f"- {item['celebrity']} | distância={item['distance']:.4f} | relative={item['relative_path']} | path={item['path']}")
