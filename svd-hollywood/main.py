import os
import kagglehub

# usar o caminho do terminal local + svd-hollywood/dataset como cache/download
cache_path = os.path.join(os.getcwd(), "svd-hollywood", "dataset")
os.environ["KAGGLEHUB_CACHE"] = cache_path

# garantir que a pasta exista antes do download
os.makedirs(cache_path, exist_ok=True)

dataset_dir = cache_path  # pasta onde o kagglehub deixará os arquivos

if os.path.isdir(dataset_dir) and any(os.scandir(dataset_dir)):
    print(f"Dataset encontrado em: {dataset_dir}")
else:
    print("Baixando dataset do Kaggle...")
    download_path = kagglehub.dataset_download("bhaveshmittal/celebrity-face-recognition-dataset")

    if not download_path:
        raise RuntimeError("Falha ao baixar dataset com kagglehub")

    download_path = os.path.abspath(download_path)
    print(f"Download concluído. Arquivos disponíveis em: {dataset_dir}")