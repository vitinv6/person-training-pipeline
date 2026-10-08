"""Entrypoint da pipeline de treinamento de modelo de detecção de pessoas."""

import argparse
import logging
from pathlib import Path

from src.config import (
    AugmentationConfig,
    DatasetConfig,
    EvaluationConfig,
    ExportConfig,
    PipelineConfig,
    TrainingConfig,
)
from src.pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de treinamento YOLOv8 para detecção de pessoas"
    )

    # Dataset
    parser.add_argument(
        "--images-dir",
        default="data/raw/images",
        help="Diretório com imagens raw (default: data/raw/images)",
    )
    parser.add_argument(
        "--labels-dir",
        default="data/raw/labels",
        help="Diretório com labels YOLO format (default: data/raw/labels)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Diretório de saída do dataset processado (default: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proporção de treino (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Proporção de validação (default: 0.2)",
    )

    # Augmentation
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Desabilita augmentation offline",
    )
    parser.add_argument(
        "--aug-multiplier",
        type=int,
        default=3,
        help="Variações por imagem no augmentation (default: 3)",
    )

    # Training
    parser.add_argument(
        "--model",
        default="yolov8n.yaml",
        help="Arquitetura YOLO (yaml) ou pesos pré-treinados (pt) (default: yolov8n.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Tamanho da imagem de entrada (default: 640)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate inicial (default: 0.01)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["SGD", "Adam", "AdamW"],
        default="SGD",
        help="Optimizer (default: SGD)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Device: '', 'cpu', 'cuda', '0' (default: auto)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers (default: 8)",
    )

    parser.add_argument("--project-dir", default="runs/train", help="Diretorio de saida dos treinamentos")
    parser.add_argument("--experiment-name", default="person_detector", help="Nome da execucao do experimento")
    parser.add_argument("--eval-dir", default="runs/eval", help="Diretorio de saida das avaliacoes")
    # Export
    parser.add_argument(
        "--export-formats",
        nargs="+",
        default=["onnx"],
        help="Formatos de export (default: onnx). Opções: onnx, torchscript, engine, coreml, openvino, tflite",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export com FP16 (half precision)",
    )

    parser.add_argument(
        "--data-yaml",
        default=None,
        help="Caminho para um data.yaml já existente",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Pula preparação do dataset (usa data.yaml existente)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Pula treinamento (requer --model-path)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Pula avaliação",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Pula export",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Caminho de modelo existente (para eval/export sem treinar)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma treinamento de checkpoint anterior",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
        if not data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml não encontrado: {data_yaml}"
            )

    # Determina se é do zero (yaml) ou fine-tuning (pt)
    is_from_scratch = args.model.endswith(".yaml")
    pretrained_weights = "" if is_from_scratch else args.model
    model_variant = args.model if is_from_scratch else "yolov8n.yaml"

    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    data_output_dir = (
        str(Path(args.data_yaml).parent)
        if args.data_yaml
        else args.output_dir
    )

    config = PipelineConfig(
        dataset=DatasetConfig(
            raw_images_dir=args.images_dir,
            raw_labels_dir=args.labels_dir,
            output_dir=data_output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
        ),
        augmentation=AugmentationConfig(
            enabled=not args.no_augment,
            multiplier=args.aug_multiplier,
        ),
        training=TrainingConfig(
            model_variant=model_variant,
            pretrained_weights=pretrained_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.img_size,
            learning_rate=args.lr,
            optimizer=args.optimizer,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            project_dir=args.project_dir,
            experiment_name=args.experiment_name,
            resume=args.resume,
        ),
        evaluation=EvaluationConfig(output_dir=args.eval_dir),
        export=ExportConfig(
            formats=args.export_formats,
            half_precision=args.half,
            image_size=args.img_size,
        ),
    )

    if args.data_yaml:
        config.dataset.output_dir = str(Path(args.data_yaml).parent)

    pipeline = TrainingPipeline(config)
    pipeline.run(
        skip_prepare=args.skip_prepare or args.data_yaml is not None,
        skip_augment=args.no_augment,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        skip_export=args.skip_export,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()


