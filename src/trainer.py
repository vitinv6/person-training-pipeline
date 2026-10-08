"""Módulo de treinamento — treina YOLOv8 do zero ou com fine-tuning."""

import logging
from pathlib import Path

from ultralytics import YOLO

from src.config import TrainingConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Treina um modelo YOLOv8 para detecção de pessoas."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._model: YOLO | None = None

    def _load_model(self) -> YOLO:
        """Carrega o modelo — do zero (yaml) ou com pesos pré-treinados."""
        if self._config.pretrained_weights:
            logger.info(
                "Carregando modelo com pesos pré-treinados: %s",
                self._config.pretrained_weights,
            )
            return YOLO(self._config.pretrained_weights)

        logger.info(
            "Criando modelo do zero a partir da arquitetura: %s",
            self._config.model_variant,
        )
        return YOLO(self._config.model_variant)

    def train(self, data_yaml: Path) -> Path:
        """Executa o treinamento completo.

        Args:
            data_yaml: Caminho para o data.yaml do dataset.

        Returns:
            Caminho do melhor modelo treinado (best.pt).
        """
        self._model = self._load_model()

        logger.info(
            "Iniciando treinamento — epochs: %d, batch: %d, img_size: %d",
            self._config.epochs,
            self._config.batch_size,
            self._config.image_size,
        )

        train_args = {
            "data": str(data_yaml),
            "epochs": self._config.epochs,
            "batch": self._config.batch_size,
            "imgsz": self._config.image_size,
            "lr0": self._config.learning_rate,
            "optimizer": self._config.optimizer,
            "momentum": self._config.momentum,
            "weight_decay": self._config.weight_decay,
            "warmup_epochs": self._config.warmup_epochs,
            "patience": self._config.patience,
            "workers": self._config.workers,
            "project": self._config.project_dir,
            "name": self._config.experiment_name,
            "exist_ok": True,
            "resume": self._config.resume,
            "verbose": True,
            # Augmentation built-in
            "mosaic": self._config.mosaic,
            "mixup": self._config.mixup,
            "copy_paste": self._config.copy_paste,
            "hsv_h": self._config.hsv_h,
            "hsv_s": self._config.hsv_s,
            "hsv_v": self._config.hsv_v,
            "flipud": self._config.flipud,
            "fliplr": self._config.fliplr,
            "degrees": self._config.degrees,
            "translate": self._config.translate,
            "scale": self._config.scale,
            "shear": self._config.shear,
            "perspective": self._config.perspective,
        }

        if self._config.device:
            train_args["device"] = self._config.device

        results = self._model.train(**train_args)

        # Usa o diretório real criado pelo Ultralytics
        save_dir = Path(self._model.trainer.save_dir)
        best_model_path = save_dir / "weights" / "best.pt"

        if best_model_path.exists():
            logger.info("Treinamento concluído — melhor modelo: %s", best_model_path)
        else:
            logger.warning(
                "best.pt não encontrado no diretório de treinamento: %s",
                best_model_path,
            )

        return best_model_path

    @property
    def model(self) -> YOLO | None:
        """Modelo treinado (disponível após train())."""
        return self._model
