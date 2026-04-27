from pathlib import Path

from modelscope import snapshot_download


BASE_DIR = Path("/home/priv/project/model")
MODEL_IDS = [
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.6-27B",
]


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for model_id in MODEL_IDS:
        model_name = model_id.split("/")[-1]
        target_dir = BASE_DIR / model_name
        print(f"Downloading {model_id} -> {target_dir}")
        snapshot_download(model_id, local_dir=str(target_dir))
        print(f"Finished {model_id}")


if __name__ == "__main__":
    main()
