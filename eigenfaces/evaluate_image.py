import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results"

face_cascade = cv2.CascadeClassifier(str(PROJECT_ROOT.parent / "src/models/haarcascade_frontalface_default.xml"))
left_eye_cascade = cv2.CascadeClassifier(str(PROJECT_ROOT.parent / "src/models/haarcascade_lefteye_2splits.xml"))
right_eye_cascade = cv2.CascadeClassifier(str(PROJECT_ROOT.parent / "src/models/haarcascade_righteye_2splits.xml"))

if face_cascade.empty():
    raise RuntimeError("Falha ao carregar cascade de face")
if left_eye_cascade.empty():
    raise RuntimeError("Falha ao carregar cascade de olho esquerdo")
if right_eye_cascade.empty():
    raise RuntimeError("Falha ao carregar cascade de olho direito")

EPS = 1e-10

EYE_SCALE = 1.05
EYE_NEIGHBORS = 3
EYE_MIN_SIZE = (8, 8)
EXPANSION_SCALE = 2.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def apply_clahe(image_gray: np.ndarray) -> np.ndarray:
    if image_gray.dtype != np.uint8:
        image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return clahe.apply(image_gray)


def apply_elliptical_mask(image_gray: np.ndarray) -> np.ndarray:
    h, w = image_gray.shape
    mask = np.zeros_like(image_gray, dtype=np.uint8)
    axes = (int(w * 0.48), int(h * 0.48))
    center = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return cv2.bitwise_and(image_gray, image_gray, mask=mask)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return vectors / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / (norm if norm > EPS else 1.0)


def expand_bbox(x, y, w, h, image_shape, scale=EXPANSION_SCALE):
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    new_w = w * scale
    new_h = h * scale
    x1 = int(round(center_x - new_w / 2.0))
    y1 = int(round(center_y - new_h / 2.0))
    x2 = int(round(center_x + new_w / 2.0))
    y2 = int(round(center_y + new_h / 2.0))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_shape[1], x2)
    y2 = min(image_shape[0], y2)
    return x1, y1, x2 - x1, y2 - y1


def preprocess_image(image_path, target_shape):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Não foi possível ler a imagem: {image_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise RuntimeError(f"Nenhuma face detectada em: {image_path}")

    x, y, w, h = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    x, y, w, h = expand_bbox(x, y, w, h, gray.shape)
    face_gray = gray[y : y + h, x : x + w]

    left_eyes = left_eye_cascade.detectMultiScale(face_gray, scaleFactor=EYE_SCALE, minNeighbors=EYE_NEIGHBORS, minSize=EYE_MIN_SIZE)
    right_eyes = right_eye_cascade.detectMultiScale(face_gray, scaleFactor=EYE_SCALE, minNeighbors=EYE_NEIGHBORS, minSize=EYE_MIN_SIZE)
    if len(left_eyes) and len(right_eyes):
        eyes = np.vstack((left_eyes, right_eyes))
    elif len(left_eyes):
        eyes = left_eyes
    else:
        eyes = right_eyes

    if len(eyes) < 2:
        left_eyes_relaxed = left_eye_cascade.detectMultiScale(face_gray, scaleFactor=1.02, minNeighbors=2, minSize=(6, 6))
        right_eyes_relaxed = right_eye_cascade.detectMultiScale(face_gray, scaleFactor=1.02, minNeighbors=2, minSize=(6, 6))
        if len(left_eyes_relaxed) and len(right_eyes_relaxed):
            eyes = np.vstack((left_eyes_relaxed, right_eyes_relaxed))
        elif len(left_eyes_relaxed):
            eyes = left_eyes_relaxed
        elif len(right_eyes_relaxed):
            eyes = right_eyes_relaxed

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: int(e[0]))
        eye1, eye2 = eyes[0], eyes[-1]
        eye_center1 = (x + eye1[0] + eye1[2] / 2.0, y + eye1[1] + eye1[3] / 2.0)
        eye_center2 = (x + eye2[0] + eye2[2] / 2.0, y + eye2[1] + eye2[3] / 2.0)

        dX = eye_center2[0] - eye_center1[0]
        dY = eye_center2[1] - eye_center1[1]
        angle = np.degrees(np.arctan2(dY, dX))
        dist = np.sqrt(dX**2 + dY**2)

        desired_dist = 0.35 * target_shape[0]
        scale = desired_dist / dist if dist > 0 else 1.0

        eyes_center = ((eye_center1[0] + eye_center2[0]) / 2.0, (eye_center1[1] + eye_center2[1]) / 2.0)
        M = cv2.getRotationMatrix2D(eyes_center, float(angle), scale)

        tX = target_shape[1] * 0.5
        tY = target_shape[0] * 0.4
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        face_gray = cv2.warpAffine(face_gray, M, (target_shape[1], target_shape[0]), flags=cv2.INTER_AREA)
    else:
        raise RuntimeError(f"Não foram encontrados dois olhos confiáveis em: {image_path}")

    if face_gray.shape != target_shape:
        face_gray = cv2.resize(face_gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

    face_gray_uint8 = cv2.normalize(face_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    face_gray_uint8 = apply_clahe(face_gray_uint8)
    face_gray_uint8 = apply_elliptical_mask(face_gray_uint8)
    gradient_map = compute_edge_gradient(face_gray_uint8)

    face_gray = face_gray_uint8.astype(np.float32) / 255.0
    gradient = gradient_map.astype(np.float32) / 255.0
    return np.concatenate([face_gray.flatten(), gradient.flatten()])


parser = argparse.ArgumentParser(description="Avalia uma imagem usando o modelo Eigenfaces")
parser.add_argument("image", type=Path, help="Caminho da imagem a ser avaliada")
parser.add_argument("--metric", choices=["l2", "cosine"], default="cosine", help="Métrica de similaridade a ser usada")
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
if args.metric == "cosine":
    query_projection = normalize_vector(query_projection)
    projected_train = normalize_rows(projections.T)
else:
    projected_train = projections.T

neighbor_count = min(3, projected_train.shape[0])
if args.metric == "l2":
    scores = np.linalg.norm(projected_train - query_projection, axis=1)
    order = np.argsort(scores)
    score_name = "distance"
else:
    similarity = projected_train.dot(query_projection)
    scores = 1.0 - similarity
    order = np.argsort(scores)
    score_name = "distance"

order = order[:neighbor_count]
neighbor_scores = scores[order]

sample_rows = sorted(label_map.items(), key=lambda item: item[1]["column"])
neighbor_labels = [sample_rows[i][1]["celebrity"] for i in order]
predicted = Counter(neighbor_labels).most_common(1)[0][0]
prediction_count = Counter(neighbor_labels)[predicted]

ranked = []
for idx, i in enumerate(order):
    relative_path, info = sample_rows[i]
    ranked.append(
        {
            "celebrity": info["celebrity"],
            "path": info["path"],
            "relative_path": relative_path,
            "column": info["column"],
            score_name: float(scores[i]),
        }
    )

best = ranked[0]

print(f"Métrica: {args.metric}")
print(f"Predição por moda (top {neighbor_count}): {predicted}")
print(f"Frequência no top {neighbor_count}: {prediction_count}")
print(f"Melhor imagem de referência: {best['path']}")
print(f"{score_name.capitalize()}: {best[score_name]:.4f}")
print("Top 5:")
for item in ranked[:5]:
    score_value = item[score_name]
    print(f"- {item['celebrity']} | {score_name}={score_value:.4f} | relative={item['relative_path']} | path={item['path']}")
