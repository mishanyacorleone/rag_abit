import os
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path("app/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit": "Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
    "deepvk/USER-bge-m3": "USER-bge-m3",
}


def main():
    for repo_id, local_name in MODELS.items():
        local_path = MODELS_DIR / local_name

        if local_path.exists() and any(local_path.iterdir()):
            logger.info(f"{repo_id} - уже скачана в {local_path}")
            continue

        logger.info(f"Скачивание {repo_id} -> {local_path}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False
        )
        logger.info(f"Скачивание завершено: {repo_id}")

    logger.info("\nВсе модели скачаны!")
    logger.info(f"Содержимое {MODELS_DIR}:")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"---- {p.name}")


if __name__ == "__main__":
    main()