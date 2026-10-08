"""Configuração centralizada da pipeline de treinamento."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuração do dataset."""

    raw_images_dir: str = "data/raw/images"
    raw_labels_dir: str = "data/raw/labels"
    output_dir: str = "data/processed"
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    image_extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )
    class_names: list[str] = field(default_factory=lambda: ["person"])
    seed: int = 42


@dataclass
class AugmentationConfig:
    """Configuração de data augmentation."""

    enabled: bool = True
    multiplier: int = 3  # Quantas variações por imagem original
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotate_limit: int = 15
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    blur_limit: int = 5
    noise_var_limit: tuple[float, float] = (10.0, 50.0)
    scale_limit: tuple[float, float] = (-0.1, 0.2)
    mosaic_prob: float = 0.0  # Mosaic é aplicado pelo YOLO, não pelo albumentations
    output_dir: str = "data/augmented"


@dataclass
class TrainingConfig:
    """Hiperparâmetros de treinamento YOLOv8."""

    model_variant: str = "yolov8n.yaml"  # Arquitetura do zero (nano)
    pretrained_weights: str = ""  # Vazio = sem pesos pré-treinados (do zero)
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    optimizer: str = "SGD"  # SGD, Adam, AdamW
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    patience: int = 20  # Early stopping
    workers: int = 8
    device: str = ""  # Auto
    project_dir: str = "runs/train"
    experiment_name: str = "person_detector"
    resume: bool = False
    # Augmentation built-in do YOLO
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.0
    fliplr: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    perspective: float = 0.0


@dataclass
class EvaluationConfig:
    """Configuração de avaliação do modelo."""

    confidence: float = 0.25
    iou_threshold: float = 0.5
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    output_dir: str = "runs/eval"


@dataclass
class ExportConfig:
    """Configuração de export do modelo."""

    formats: list[str] = field(default_factory=lambda: ["onnx"])
    image_size: int = 640
    half_precision: bool = False  # FP16
    dynamic_batch: bool = False
    simplify_onnx: bool = True
    output_dir: str = "runs/export"


@dataclass
class PipelineConfig:
    """Configuração completa da pipeline de treinamento."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def data_yaml_path(self) -> Path:
        """Caminho do data.yaml gerado para o YOLO."""
        return Path(self.dataset.output_dir) / "data.yaml"
