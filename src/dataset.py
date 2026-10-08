"""Módulo de preparação de dataset — organiza imagens e labels no formato YOLO."""

import logging
import shutil
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from src.config import DatasetConfig

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Organiza imagens raw + labels no formato YOLO (train/val/test splits)."""

    def __init__(self, config: DatasetConfig) -> None:
        self._config = config

    def prepare(self) -> Path:
        """Executa a preparação completa do dataset.

        Returns:
            Caminho do data.yaml gerado.
        """
        logger.info("Preparando dataset...")

        image_paths = self._collect_images()
        if not image_paths:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada em: {self._config.raw_images_dir}"
            )

        logger.info("Imagens encontradas: %d", len(image_paths))

        self._validate_labels(image_paths)

        train_imgs, val_imgs, test_imgs = self._split(image_paths)
        logger.info(
            "Split — train: %d, val: %d, test: %d",
            len(train_imgs),
            len(val_imgs),
            len(test_imgs),
        )

        output_dir = Path(self._config.output_dir)
        self._create_structure(output_dir)
        self._copy_files(train_imgs, output_dir / "train")
        self._copy_files(val_imgs, output_dir / "val")
        self._copy_files(test_imgs, output_dir / "test")

        data_yaml_path = self._generate_data_yaml(output_dir)
        logger.info("Dataset preparado — data.yaml: %s", data_yaml_path)
        return data_yaml_path

    def _collect_images(self) -> list[Path]:
        """Coleta todos os arquivos de imagem no diretório raw."""
        raw_dir = Path(self._config.raw_images_dir)
        if not raw_dir.exists():
            return []
        images = []
        for ext in self._config.image_extensions:
            images.extend(raw_dir.glob(f"*{ext}"))
            images.extend(raw_dir.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    def _validate_labels(self, image_paths: list[Path]) -> None:
        """Verifica se cada imagem tem um label correspondente."""
        labels_dir = Path(self._config.raw_labels_dir)
        missing = []
        for img_path in image_paths:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                missing.append(img_path.name)

        if missing:
            logger.warning(
                "%d imagens sem label correspondente (serão incluídas como negativos): %s",
                len(missing),
                missing[:5],
            )

    def _split(
        self, image_paths: list[Path]
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Divide imagens em train/val/test."""
        train_imgs, temp_imgs = train_test_split(
            image_paths,
            train_size=self._config.train_ratio,
            random_state=self._config.seed,
        )

        relative_val = self._config.val_ratio / (
            self._config.val_ratio + self._config.test_ratio
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            train_size=relative_val,
            random_state=self._config.seed,
        )

        return train_imgs, val_imgs, test_imgs

    def _create_structure(self, output_dir: Path) -> None:
        """Cria a estrutura de diretórios YOLO."""
        for split in ("train", "val", "test"):
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def _copy_files(self, image_paths: list[Path], split_dir: Path) -> None:
        """Copia imagens e labels para o diretório do split."""
        labels_dir = Path(self._config.raw_labels_dir)
        for img_path in image_paths:
            # Copia imagem
            shutil.copy2(img_path, split_dir / "images" / img_path.name)
            # Copia label se existir
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, split_dir / "labels" / label_path.name)
            else:
                # Cria label vazio (imagem negativa — sem objetos)
                (split_dir / "labels" / f"{img_path.stem}.txt").touch()

    def _generate_data_yaml(self, output_dir: Path) -> Path:
        """Gera o data.yaml no formato esperado pelo Ultralytics."""
        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self._config.class_names),
            "names": self._config.class_names,
        }

        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        return yaml_path
