import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


EPS = 1e-10

def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return vectors / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / (norm if norm > EPS else 1.0)


def compute_metrics(train_labels, validation_items, projections, X_bar, U, metric="cosine", k_components=None, n_neighbors=1, exclude_self=False):
    results = []
    correct_count = 0
    if k_components is None:
        k_components = projections.shape[0]
    if k_components <= 0 or k_components > projections.shape[0]:
        raise ValueError(f"k_components must be between 1 and {projections.shape[0]}, got {k_components}")

    projected_train = projections[:k_components].T
    U_k = U[:, :k_components]
    neighbor_count = n_neighbors

    if metric == "cosine":
        projected_train = normalize_rows(projected_train)

    for relative_path, info in validation_items:
        val_path = Path(info["npy_path"])
        if not val_path.exists():
            raise RuntimeError(f"Arquivo .npy de validação não encontrado: {val_path}")

        query = np.load(val_path).astype(np.float32).flatten()
        query_centered = query - X_bar
        query_projection = U_k.T @ query_centered

        if metric == "cosine":
            query_projection = normalize_vector(query_projection)

        if exclude_self and "column" in info:
            self_index = info["column"]
            mask = np.ones(projected_train.shape[0], dtype=bool)
            if 0 <= self_index < projected_train.shape[0]:
                mask[self_index] = False
            masked_train = projected_train[mask]
            if metric == "cosine":
                scores = 1.0 - (masked_train @ query_projection)
            else:
                scores = np.linalg.norm(masked_train - query_projection, axis=1)
            order = np.argsort(scores)[:neighbor_count]
            actual_indices = np.flatnonzero(mask)[order]
        else:
            if metric == "cosine":
                scores = 1.0 - (projected_train @ query_projection)
            else:
                scores = np.linalg.norm(projected_train - query_projection, axis=1)
            order = np.argsort(scores)[:neighbor_count]
            actual_indices = order

        score_value = float(scores[order[0]])
        top_labels = [train_labels[i] for i in actual_indices]
        predicted = Counter(top_labels).most_common(1)[0][0]
        actual = info["celebrity"]
        hit = predicted == actual
        correct_count += int(hit)

        results.append(
            {
                "relative_path": relative_path,
                "celebrity": actual,
                "prediction": predicted,
                "hit": int(hit),
                "score": score_value,
                "top_labels": top_labels,
            }
        )

    total = len(results)
    accuracy = correct_count / total if total else 0.0
    summary = {
        "total": total,
        "correct": correct_count,
        "accuracy": accuracy,
    }

    return {
        "metric": metric,
        "k_components": k_components,
        "n_neighbors": n_neighbors,
        "summary": summary,
        "total": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "results": results,
    }


def search_hyperparameters(train_labels, validation_items, projections, X_bar, U, metric="l2", k_values=None, neighbor_values=None, k_max=None):
    if k_max is None:
        k_max = projections.shape[0]
    k_max = min(max(1, int(k_max)), projections.shape[0])

    if k_values is None:
        fractions = [0.10, 0.25, 0.75, 1.0]
        k_values = sorted({max(1, int(round(k_max * f))) for f in fractions})
    if neighbor_values is None:
        neighbor_values = [1, 2, 3]

    search_results = []
    k_values = [k for k in sorted(set(k_values)) if 1 <= k <= projections.shape[0]]
    neighbor_values = [n for n in sorted(set(neighbor_values)) if n >= 1]

    for k in k_values:
        for n in neighbor_values:
            metrics = compute_metrics(
                train_labels,
                validation_items,
                projections,
                X_bar,
                U,
                metric=metric,
                k_components=k,
                n_neighbors=n,
            )
            accuracy = metrics["accuracy"]
            print(f"evaluating {metric}: k={k:3d}, n_neighbors={n:2d}, accuracy={accuracy:.4f}, correct={metrics['correct']}/{metrics['total']}")
            search_results.append(
                {
                    "metric": metric,
                    "k_components": k,
                    "n_neighbors": n,
                    "accuracy": accuracy,
                    "total": metrics["total"],
                    "correct": metrics["correct"],
                }
            )

    if not search_results:
        raise RuntimeError("Nenhum valor válido de k_components ou n_neighbors fornecido para grid search")

    best_result = max(search_results, key=lambda item: item["accuracy"])
    print(f"best {metric}: k={best_result['k_components']}, n_neighbors={best_result['n_neighbors']}, accuracy={best_result['accuracy']:.4f}")
    return best_result, search_results


def save_grid_search(search_results, output_dir: Path, metric: str):
    grid_path = output_dir / f"validation_hyperparameter_search_{metric}.json"
    with grid_path.open("w", encoding="utf-8") as f:
        json.dump({"results": search_results}, f, ensure_ascii=False, indent=2)
    print(f"Grid search salvo em: {grid_path}")


def print_summary(metrics):
    accuracy_pct = metrics["accuracy"] * 100.0
    print(f"Validation Metrics ({metrics['metric']})")
    print("------------------")
    print(f"Total examples: {metrics['total']}")
    print(f"Correct predictions: {metrics['correct']}")
    print(f"Accuracy: {accuracy_pct:.2f}%")
    print("")


def main():
    label_map_train = load_json(OUTPUT_DIR / "label_map_train.json")
    label_map_validation = load_json(OUTPUT_DIR / "label_map_validation.json")
    X_bar = np.load(OUTPUT_DIR / "X_bar.npy")
    U = np.load(OUTPUT_DIR / "U.npy")
    projections = np.load(OUTPUT_DIR / "projections.npy")

    train_items = sorted(label_map_train.items(), key=lambda item: item[1]["column"])
    train_labels = [info["celebrity"] for _, info in train_items]
    validation_items = sorted(label_map_validation.items())
    svd_metadata = load_json(OUTPUT_DIR / "svd_metadata.json")
    k_max = svd_metadata.get("k_max", projections.shape[0])

    metric = "cosine"
    print(f"\n=== Grid search for voting-based k on {metric} distance ===")
    best_result, search_results = search_hyperparameters(
        train_labels,
        validation_items,
        projections,
        X_bar,
        U,
        metric=metric,
        k_max=k_max,
    )
    save_grid_search(search_results, OUTPUT_DIR, metric)
    print(
        f"Selected best voting k from k_max={k_max}: k={best_result['k_components']}, n_neighbors={best_result['n_neighbors']}, accuracy={best_result['accuracy']:.4f}\n"
    )


if __name__ == "__main__":
    main()
