"""Módulo de avaliação — executa validação e gera métricas do modelo treinado."""

import logging
from pathlib import Path

from ultralytics import YOLO

from src.config import EvaluationConfig

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Avalia o modelo treinado no dataset de teste/validação."""

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    def evaluate(self, model_path: Path, data_yaml: Path) -> dict:
        """Executa avaliação no dataset de teste.

        Args:
            model_path: Caminho do best.pt treinado.
            data_yaml: Caminho do data.yaml.

        Returns:
            Dicionário com métricas (mAP50, mAP50-95, precision, recall).
        """
        logger.info("Avaliando modelo: %s", model_path)

        model = YOLO(str(model_path))

        results = model.val(
            data=str(data_yaml),
            conf=self._config.confidence,
            iou=self._config.iou_threshold,
            split="val",
            project=self._config.output_dir,
            name="eval",
            exist_ok=True,
            save_json=self._config.save_predictions,
            plots=self._config.save_confusion_matrix,
            verbose=True,
        )

        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }

        logger.info("=== Resultados da Avaliação ===")
        logger.info("  mAP@0.5:      %.4f", metrics["mAP50"])
        logger.info("  mAP@0.5:0.95: %.4f", metrics["mAP50_95"])
        logger.info("  Precision:     %.4f", metrics["precision"])
        logger.info("  Recall:        %.4f", metrics["recall"])

        return metrics

    def predict_samples(
        self, model_path: Path, images_dir: str, output_dir: str | None = None
    ) -> None:
        """Roda inferência em imagens de amostra e salva resultados visuais.

        Args:
            model_path: Caminho do modelo.
            images_dir: Diretório com imagens para predição.
            output_dir: Diretório de saída (default: runs/eval/predictions).
        """
        save_dir = output_dir or str(Path(self._config.output_dir) / "predictions")

        logger.info("Rodando predição em amostras: %s", images_dir)

        model = YOLO(str(model_path))
        model.predict(
            source=images_dir,
            conf=self._config.confidence,
            iou=self._config.iou_threshold,
            save=True,
            project=save_dir,
            name="samples",
            exist_ok=True,
        )

        logger.info("Predições salvas em: %s/samples", save_dir)
