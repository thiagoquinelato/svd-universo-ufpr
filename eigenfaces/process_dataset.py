import json
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "datasets"
DATASET_DIR = CACHE_DIR / "dataset_celeb_br"
PROCESSED_DIR = CACHE_DIR / "dataset_celeb_br_processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VALIDATION_DIR = PROCESSED_DIR / "validation"
OUTPUT_DIR = PROJECT_ROOT / "results"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

face_cascade = cv2.CascadeClassifier(str(PROJECT_ROOT.parent / "src/models/haarcascade_frontalface_default.xml"))
left_eye_cascade = cv2.CascadeClassifier(str(PROJECT_ROOT.parent / "src/models/haarcascade_lefteye_2splits.xml"))
right_eye_cascade = cv2.CascadeClassifier(str(PROJECT_ROOT.parent / "src/models/haarcascade_righteye_2splits.xml"))

if face_cascade.empty():
    raise RuntimeError("Falha ao carregar cascade de face")
if left_eye_cascade.empty():
    raise RuntimeError("Falha ao carregar cascade de olho esquerdo")
if right_eye_cascade.empty():
    raise RuntimeError("Falha ao carregar cascade de olho direito")


def normalize_map(image_float):
    mean = image_float.mean()
    std = image_float.std()
    std = std if std > 1e-6 else 1.0
    return (image_float - mean) / std


def normalize_display(image_float):
    display_image = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)
    return display_image.astype(np.uint8)


def load_metadata():
    metadata_path = DATASET_DIR / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("metadata.json não encontrado! Execute download_dataset.py primeiro.")

    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir, relative_path):
    candidate = base_dir / relative_path
    if candidate.exists():
        return candidate
    matches = list(base_dir.rglob(relative_path.name))
    return matches[0] if matches else candidate


def is_dataset_processed() -> bool:
    label_map_file = OUTPUT_DIR / "label_map_train.json"
    if not label_map_file.exists():
        return False
    if not TRAIN_DIR.exists() or not VALIDATION_DIR.exists():
        return False
    if not any(TRAIN_DIR.iterdir()):
        return False
    if not any(VALIDATION_DIR.iterdir()):
        return False
    return True


IMAGE_SIZE = (128, 128)
FACE_SCALE = 1.1
FACE_NEIGHBORS = 5
FACE_MIN_SIZE = (30, 30)
EYE_SCALE = 1.05
EYE_NEIGHBORS = 3
EYE_MIN_SIZE = (8, 8)
EXPANSION_SCALE = 2.0  # Expande a janela do rosto de forma moderada
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


def expand_bbox(x, y, w, h, image_shape, scale=EXPANSION_SCALE):
    cy, cx = image_shape[0] / 2.0, image_shape[1] / 2.0
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


def compute_edge_gradient(image_gray_uint8: np.ndarray) -> np.ndarray:
    sobel_x = cv2.Sobel(image_gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    return gradient


def process_image(image_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Não foi possível ler a imagem: {image_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=FACE_SCALE, minNeighbors=FACE_NEIGHBORS, minSize=FACE_MIN_SIZE)
    if len(faces) == 0:
        raise RuntimeError("Nenhuma face detectada")

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
        eye_center1 = (eye1[0] + eye1[2] / 2.0, eye1[1] + eye1[3] / 2.0)
        eye_center2 = (eye2[0] + eye2[2] / 2.0, eye2[1] + eye2[3] / 2.0)
        
        dX = eye_center2[0] - eye_center1[0]
        dY = eye_center2[1] - eye_center1[1]
        angle = np.degrees(np.arctan2(dY, dX))
        dist = np.sqrt(dX**2 + dY**2)
        
        # Alinhamento Canônico (Proporções fixas no frame de destino)
        desired_dist = 0.25 * IMAGE_SIZE[0]  # Distância dos olhos = 25% da imagem para mais margem
        scale = desired_dist / dist if dist > 0 else 1.0

        eyes_center = ((eye_center1[0] + eye_center2[0]) / 2.0, (eye_center1[1] + eye_center2[1]) / 2.0)
        M = cv2.getRotationMatrix2D(eyes_center, float(angle), scale)
        
        # Desloca o centro dos olhos para a posição padrão (x=50%, y=45%)
        tX = IMAGE_SIZE[1] * 0.5
        tY = IMAGE_SIZE[0] * 0.45
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        face_gray = cv2.warpAffine(face_gray, M, (IMAGE_SIZE[1], IMAGE_SIZE[0]), flags=cv2.INTER_AREA)
    else:
        raise RuntimeError("Não foram encontrados dois olhos confiáveis para alinhamento")

    if face_gray.shape != IMAGE_SIZE:
        face_gray = cv2.resize(face_gray, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)

    face_gray_uint8 = cv2.normalize(face_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    face_gray_uint8 = apply_clahe(face_gray_uint8)
    face_gray_uint8 = apply_elliptical_mask(face_gray_uint8)
    gradient_map = compute_edge_gradient(face_gray_uint8)

    face_gray = face_gray_uint8.astype(np.float32) / 255.0
    gradient = gradient_map.astype(np.float32) / 255.0
    return np.concatenate([face_gray.flatten(), gradient.flatten()])


def main():
    if is_dataset_processed():
        print(f"Dataset já processado em: {OUTPUT_DIR}. Pulando process_dataset.")
        return

    dataset_metadata = load_metadata()
    base_dir = DATASET_DIR
    raw_images = dataset_metadata.get("images", [])
    raw_image_count = len(raw_images)
    raw_artists = sorted({item["celebrity"] for item in raw_images})
    print(f"Dataset original: {len(raw_artists)} artistas, {raw_image_count} imagens")
    samples = []
    skipped_count = 0

    for item in raw_images:
        rel_path = Path(item["path"])
        image_path = resolve_path(base_dir, rel_path)
        if not image_path.exists():
            print(f"Aviso: caminho de imagem não encontrado no metadata: {item['path']}")
            skipped_count += 1
            continue

        try:
            processed = process_image(image_path)
        except Exception as exc:
            print(f"Aviso: falha ao processar {image_path}: {exc}")
            skipped_count += 1
            continue

        expected_length = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 2
        if processed.size != expected_length:
            print(f"Aviso: imagem processada com forma inesperada {processed.shape}, ignorando: {image_path}")
            skipped_count += 1
            continue

        samples.append((item["celebrity"], rel_path, processed))

    if not samples:
        raise RuntimeError("Nenhuma imagem válida foi processada")

    processed_image_count = len(samples)
    processed_artists = sorted({celebrity for celebrity, _, _ in samples})
    print(f"Dataset processado: {len(processed_artists)} artistas, {processed_image_count} imagens válidas")

    samples_by_celebrity = {}
    for celebrity, rel_path, processed in samples:
        samples_by_celebrity.setdefault(celebrity, []).append((rel_path, processed))

    train_samples = []
    validation_samples = []
    rng = random.Random()
    for celebrity, celeb_samples in samples_by_celebrity.items():
        rng.shuffle(celeb_samples)
        validation_count = max(1, int(round(len(celeb_samples) * 0.1))) if len(celeb_samples) > 1 else 0
        train_count = len(celeb_samples) - validation_count
        train_samples.extend([(celebrity, rel_path, processed) for rel_path, processed in celeb_samples[:train_count]])
        validation_samples.extend([(celebrity, rel_path, processed) for rel_path, processed in celeb_samples[train_count:]])

    label_map_train = {}
    grayscale_length = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    for column_index, (celebrity, rel_path, processed) in enumerate(train_samples):
        output_name = rel_path.name
        image_path = TRAIN_DIR / output_name
        display_image = normalize_display(processed[:grayscale_length].reshape(IMAGE_SIZE))
        cv2.imwrite(str(image_path), display_image)
        npy_path = image_path.with_suffix(".npy")
        np.save(str(npy_path), processed.astype(np.float32))

        label_map_train[output_name] = {
            "column": column_index,
            "path": str(image_path),
            "npy_path": str(npy_path),
            "celebrity": celebrity,
            "split": "train",
        }

    label_map_validation = {}
    grayscale_length = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    for celebrity, rel_path, processed in validation_samples:
        output_name = rel_path.name
        image_path = VALIDATION_DIR / output_name
        display_image = normalize_display(processed[:grayscale_length].reshape(IMAGE_SIZE))
        cv2.imwrite(str(image_path), display_image)
        npy_path = image_path.with_suffix(".npy")
        np.save(str(npy_path), processed.astype(np.float32))

        label_map_validation[output_name] = {
            "path": str(image_path),
            "npy_path": str(npy_path),
            "celebrity": celebrity,
            "split": "validation",
        }

    with open(OUTPUT_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_train, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "label_map_train.json", "w", encoding="utf-8") as f:
        json.dump(label_map_train, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "label_map_validation.json", "w", encoding="utf-8") as f:
        json.dump(label_map_validation, f, ensure_ascii=False, indent=2)

    metadata = {
        "image_shape": [IMAGE_SIZE[1], IMAGE_SIZE[0]],
        "resolution": [IMAGE_SIZE[1], IMAGE_SIZE[0]],
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "processed_sample_count": len(samples),
        "skipped_count": skipped_count,
        "split_ratio": [0.9, 0.1],
        "split_strategy": "per-celebrity-90-10",
        "train_dir": str(TRAIN_DIR),
        "validation_dir": str(VALIDATION_DIR),
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Processamento concluído: {len(train_samples)} treino, {len(validation_samples)} validação")
    print(f"Treino salvo em: {TRAIN_DIR}")
    print(f"Validação salva em: {VALIDATION_DIR}")
    print(f"Resultados salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
