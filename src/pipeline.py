"""Pipeline principal — orquestra todo o fluxo de treinamento."""

import logging
import shutil
from pathlib import Path

from src.augmentation import DataAugmentor
from src.config import PipelineConfig
from src.dataset import DatasetPreparer
from src.evaluator import ModelEvaluator
from src.exporter import ModelExporter
from src.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orquestra: preparação → augmentation → treinamento → avaliação → export."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._dataset_preparer = DatasetPreparer(config.dataset)
        self._augmentor = DataAugmentor(config.augmentation)
        self._trainer = ModelTrainer(config.training)
        self._evaluator = ModelEvaluator(config.evaluation)
        self._exporter = ModelExporter(config.export)

    def run(
        self,
        skip_prepare: bool = False,
        skip_augment: bool = False,
        skip_train: bool = False,
        skip_eval: bool = False,
        skip_export: bool = False,
        model_path: str | None = None,
    ) -> None:
        """Executa a pipeline completa (ou etapas selecionadas).

        Args:
            skip_prepare: Pula preparação de dataset (usa data.yaml existente).
            skip_augment: Pula augmentation offline.
            skip_train: Pula treinamento (requer model_path).
            skip_eval: Pula avaliação.
            skip_export: Pula export.
            model_path: Caminho de modelo existente (para eval/export sem treinar).
        """
        logger.info("=" * 60)
        logger.info("PIPELINE DE TREINAMENTO — DETECÇÃO DE PESSOAS")
        logger.info("=" * 60)

        # 1. Preparação do dataset
        data_yaml = self._config.data_yaml_path
        if not skip_prepare:
            data_yaml = self._step_prepare_dataset()
        else:
            logger.info("[SKIP] Preparação de dataset")
            if not data_yaml.exists():
                raise FileNotFoundError(
                    f"data.yaml não encontrado: {data_yaml}. Rode sem --skip-prepare."
                )

        # 2. Data augmentation offline
        if not skip_augment:
            self._step_augment(data_yaml)
        else:
            logger.info("[SKIP] Data augmentation")

        # 3. Treinamento
        best_model = Path(model_path) if model_path else None
        if not skip_train:
            best_model = self._step_train(data_yaml)
        else:
         logger.info("[SKIP] Treinamento")
         if not skip_eval and not skip_export:
             if best_model is None or not best_model.exists():
                 raise FileNotFoundError(
                     "model_path obrigatório quando skip_train=True"
                 )

        # 4. Avaliação
        if not skip_eval and best_model is not None:
            self._step_evaluate(best_model, data_yaml)
        else:
            logger.info("[SKIP] Avaliação")

        # 5. Export
        if not skip_export and best_model is not None:
            self._step_export(best_model)
        else:
            logger.info("[SKIP] Export")

        logger.info("=" * 60)
        logger.info("PIPELINE CONCLUÍDA")
        logger.info("=" * 60)

    def _step_prepare_dataset(self) -> Path:
        """Etapa 1: Preparação do dataset."""
        logger.info("-" * 40)
        logger.info("ETAPA 1/5: Preparação do dataset")
        logger.info("-" * 40)
        return self._dataset_preparer.prepare()

    def _step_augment(self, data_yaml: Path) -> None:
        """Etapa 2: Data augmentation offline."""
        logger.info("-" * 40)
        logger.info("ETAPA 2/5: Data augmentation")
        logger.info("-" * 40)

        processed_dir = Path(self._config.dataset.output_dir)
        train_images = str(processed_dir / "train" / "images")
        train_labels = str(processed_dir / "train" / "labels")

        self._augmentor.augment_dataset(train_images, train_labels)

        # Copia augmented para o train set
        if self._config.augmentation.enabled:
            aug_images = Path(self._config.augmentation.output_dir) / "images"
            aug_labels = Path(self._config.augmentation.output_dir) / "labels"
            if aug_images.exists():
                dest_images = processed_dir / "train" / "images"
                dest_labels = processed_dir / "train" / "labels"
                for img in aug_images.iterdir():
                    shutil.copy2(img, dest_images / img.name)
                for lbl in aug_labels.iterdir():
                    shutil.copy2(lbl, dest_labels / lbl.name)
                logger.info("Augmented data copiado para train set.")

    def _step_train(self, data_yaml: Path) -> Path:
        """Etapa 3: Treinamento."""
        logger.info("-" * 40)
        logger.info("ETAPA 3/5: Treinamento")
        logger.info("-" * 40)
        return self._trainer.train(data_yaml)

    def _step_evaluate(self, model_path: Path, data_yaml: Path) -> dict:
        """Etapa 4: Avaliação."""
        logger.info("-" * 40)
        logger.info("ETAPA 4/5: Avaliação")
        logger.info("-" * 40)
        return self._evaluator.evaluate(model_path, data_yaml)

    def _step_export(self, model_path: Path) -> list[Path]:
        """Etapa 5: Export."""
        logger.info("-" * 40)
        logger.info("ETAPA 5/5: Export")
        logger.info("-" * 40)
        return self._exporter.export(model_path)
