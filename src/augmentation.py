"""Módulo de data augmentation offline usando Albumentations."""

import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from src.config import AugmentationConfig

logger = logging.getLogger(__name__)


class DataAugmentor:
    """Aplica augmentation offline nas imagens de treino, gerando variações extras."""

    def __init__(self, config: AugmentationConfig) -> None:
        self._config = config
        self._transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        """Monta o pipeline de augmentation com Albumentations."""
        return A.Compose(
            [
                A.HorizontalFlip(p=self._config.horizontal_flip_prob),
                A.VerticalFlip(p=self._config.vertical_flip_prob),
                A.Rotate(limit=self._config.rotate_limit, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=self._config.brightness_limit,
                    contrast_limit=self._config.contrast_limit,
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=self._config.blur_limit, p=0.3),
                A.GaussNoise(var_limit=self._config.noise_var_limit, p=0.3),
                A.Affine(
                    scale=(1.0 + self._config.scale_limit[0], 1.0 + self._config.scale_limit[1]),
                    p=0.4,
                ),
                A.CLAHE(clip_limit=4.0, p=0.2),
                A.RandomShadow(p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3,
                clip=True,
            ),
        )

    def augment_dataset(self, train_images_dir: str, train_labels_dir: str) -> None:
        """Aplica augmentation em todas as imagens de treino.

        Gera `multiplier` variações por imagem original e salva no output_dir.
        """
        if not self._config.enabled:
            logger.info("Augmentation desabilitado — pulando.")
            return

        images_dir = Path(train_images_dir)
        labels_dir = Path(train_labels_dir)
        output_images = Path(self._config.output_dir) / "images"
        output_labels = Path(self._config.output_dir) / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        image_files = sorted(images_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]

        total_generated = 0

        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Não foi possível ler: %s", img_path)
                continue

            bboxes, class_labels = self._load_yolo_labels(
                labels_dir / f"{img_path.stem}.txt"
            )

            for i in range(self._config.multiplier):
                aug_image, aug_bboxes, aug_labels = self._apply_transform(
                    image, bboxes, class_labels
                )

                aug_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                cv2.imwrite(str(output_images / aug_name), aug_image)
                self._save_yolo_labels(
                    output_labels / f"{img_path.stem}_aug{i}.txt",
                    aug_bboxes,
                    aug_labels,
                )
                total_generated += 1

        logger.info(
            "Augmentation completo — %d variações geradas em %s",
            total_generated,
            self._config.output_dir,
        )

    def _load_yolo_labels(
        self, label_path: Path
    ) -> tuple[list[list[float]], list[int]]:
        """Carrega labels no formato YOLO (class cx cy w h)."""
        bboxes: list[list[float]] = []
        class_labels: list[int] = []

        if not label_path.exists():
            return bboxes, class_labels

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append([cx, cy, w, h])
                class_labels.append(cls_id)

        return bboxes, class_labels

    def _apply_transform(
        self,
        image: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[int],
    ) -> tuple[np.ndarray, list[list[float]], list[int]]:
        """Aplica a transformação a uma imagem + bboxes."""
        transformed = self._transform(
            image=image, bboxes=bboxes, class_labels=class_labels
        )
        return (
            transformed["image"],
            transformed["bboxes"],
            transformed["class_labels"],
        )

    def _save_yolo_labels(
        self, path: Path, bboxes: list[list[float]], class_labels: list[int]
    ) -> None:
        """Salva labels no formato YOLO."""
        with open(path, "w") as f:
            for bbox, cls_id in zip(bboxes, class_labels):
                cx, cy, w, h = bbox
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

