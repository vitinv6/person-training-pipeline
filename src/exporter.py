"""Módulo de export — converte modelo treinado para formatos de deploy."""

import logging
from pathlib import Path

from ultralytics import YOLO

from src.config import ExportConfig

logger = logging.getLogger(__name__)

# Formatos suportados pelo Ultralytics
SUPPORTED_FORMATS = {
    "onnx": "ONNX",
    "torchscript": "TorchScript",
    "engine": "TensorRT",
    "coreml": "CoreML",
    "openvino": "OpenVINO",
    "tflite": "TFLite",
    "ncnn": "NCNN",
}


class ModelExporter:
    """Exporta modelo YOLOv8 treinado para formatos de deploy (ONNX, TensorRT, etc.)."""

    def __init__(self, config: ExportConfig) -> None:
        self._config = config

    def export(self, model_path: Path) -> list[Path]:
        """Exporta o modelo para todos os formatos configurados.

        Args:
            model_path: Caminho do best.pt treinado.

        Returns:
            Lista de caminhos dos modelos exportados.
        """
        logger.info("Exportando modelo: %s", model_path)
        model = YOLO(str(model_path))

        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_paths: list[Path] = []

        for fmt in self._config.formats:
            if fmt not in SUPPORTED_FORMATS:
                logger.warning(
                    "Formato '%s' não suportado — opções: %s",
                    fmt,
                    list(SUPPORTED_FORMATS.keys()),
                )
                continue

            logger.info("Exportando para %s (%s)...", fmt, SUPPORTED_FORMATS[fmt])

            export_args: dict = {
                "format": fmt,
                "imgsz": self._config.image_size,
                "half": self._config.half_precision,
                "dynamic": self._config.dynamic_batch,
            }

            if fmt == "onnx":
                export_args["simplify"] = self._config.simplify_onnx

            exported_path = model.export(**export_args)

            if exported_path:
                result_path = Path(exported_path)
                exported_paths.append(result_path)
                logger.info("  → Exportado: %s", result_path)
            else:
                logger.error("  → Falha ao exportar para %s", fmt)

        logger.info("Export concluído — %d modelos gerados.", len(exported_paths))
        return exported_paths
