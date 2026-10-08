# Person Training Pipeline

Pipeline em Python para experimentos de **detecção de pessoas em imagens de transporte público utilizando YOLOv8**, desenvolvido como parte do Trabalho de Conclusão de Curso.

O projeto foi estruturado para permitir a organização, execução e avaliação de diferentes experimentos de detecção, mantendo separadas as configurações, modelos e resultados de cada abordagem.

---

## Objetivo

Investigar o uso de **visão computacional e detecção de objetos** para apoiar a estimativa do nível de ocupação em veículos de transporte público.

A abordagem utiliza modelos YOLOv8 e compara diferentes estratégias de detecção, mantendo os experimentos independentes para possibilitar uma análise adequada dos resultados.

---

## Experimentos

### Exp1 — Baseline

Experimento de referência utilizando o **YOLOv8n pré-treinado**, sem treinamento específico no dataset do projeto.

O objetivo é estabelecer um baseline para comparar o comportamento de um detector previamente treinado com os modelos posteriormente adaptados ao problema.

- Modelo: YOLOv8n pré-treinado
- Dataset de treinamento específico: não utilizado
- Classe avaliada: `person`
- Finalidade: baseline de comparação

---

### Exp2 — Bus Occupancy

Experimento voltado à **detecção de pessoas em imagens de ônibus**, utilizando um dataset específico para o problema de ocupação.

O modelo foi treinado para detectar pessoas nas imagens utilizadas pelo projeto.

Estrutura associada:

```text
data/datasets/exp2_bus_occupancy/
models/trained/exp2_bus_occupancy/
results/exp2_bus_occupancy/
```

O melhor modelo treinado é armazenado em:

```text
models/trained/exp2_bus_occupancy/best.pt
```

---

### Exp3 — Head Only

Experimento complementar baseado na **detecção das cabeças das pessoas**.

A proposta é avaliar uma representação alternativa para a estimativa de ocupação, especialmente em situações nas quais o corpo completo das pessoas pode estar parcialmente oculto ou sobreposto.

Estrutura associada:

```text
data/datasets/exp3_head_only/
models/trained/exp3_head_only/
results/exp3_head_only/
```

O melhor modelo treinado é armazenado em:

```text
models/trained/exp3_head_only/best.pt
```

Os resultados do Exp3 são mantidos separados dos resultados do Exp2 para evitar mistura entre os experimentos.

---

## Estrutura do projeto

```text
person-training-pipeline/
│
├── data/
│   ├── augmented/
│   │   ├── images/
│   │   └── labels/
│   │
│   └── datasets/
│       ├── exp2_bus_occupancy/
│       └── exp3_head_only/
│
├── docs/
│
├── models/
│   ├── pretrained/
│   └── trained/
│       ├── exp2_bus_occupancy/
│       └── exp3_head_only/
│
├── results/
│   ├── exp2_bus_occupancy/
│   └── exp3_head_only/
│
├── runs/
│   ├── detect/
│   ├── exp1_baseline/
│   ├── exp2_bus_occupancy/
│   └── exp3_bus_head/
│
├── src/
│   ├── __init__.py
│   ├── augmentation.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── exporter.py
│   ├── pipeline.py
│   └── trainer.py
│
├── test_images/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

> Os diretórios de dados e artefatos de treinamento podem existir localmente, mas nem todos são versionados no GitHub.

---

## Pipeline

O projeto utiliza uma estrutura modular para organizar as principais etapas:

```text
Dataset
   ↓
Preparação / Augmentation
   ↓
Treinamento YOLOv8
   ↓
Validação / Avaliação
   ↓
Predições
   ↓
Análise dos resultados
```

Os módulos principais estão localizados em `src/`:

| Arquivo | Função |
|---|---|
| `config.py` | Configurações da pipeline |
| `dataset.py` | Preparação e organização dos datasets |
| `augmentation.py` | Processamento de data augmentation |
| `trainer.py` | Treinamento dos modelos |
| `evaluator.py` | Avaliação dos modelos |
| `exporter.py` | Exportação dos modelos |
| `pipeline.py` | Orquestração das etapas |

O arquivo `main.py` funciona como ponto de entrada da aplicação.

---

## Dados

O projeto utiliza datasets anotados no formato YOLO.

O conjunto de augmentation atualmente presente em:

```text
data/augmented/
```

possui:

- **1.767 imagens**
- **1.767 arquivos de anotação**

Existe correspondência entre as imagens e suas respectivas anotações.

Os datasets completos não são versionados no repositório GitHub por causa do volume de arquivos. Eles permanecem disponíveis localmente para execução e desenvolvimento.

---

## Modelos

Os pesos dos modelos são organizados de acordo com sua finalidade:

```text
models/
├── pretrained/
│   └── yolov8n.pt
│
└── trained/
    ├── exp2_bus_occupancy/
    │   └── best.pt
    │
    └── exp3_head_only/
        └── best.pt
```

Essa separação evita confundir pesos pré-treinados com modelos obtidos durante os experimentos.

---

## Resultados

Os resultados são separados por experimento:

```text
results/
├── exp2_bus_occupancy/
└── exp3_head_only/
```

Essa organização é utilizada para garantir que métricas, predições e imagens de avaliação de diferentes experimentos não sejam misturadas.

Os artefatos históricos dos treinamentos permanecem organizados em:

```text
runs/
```

incluindo registros relacionados aos três experimentos.

---

## Avaliação

Entre as métricas utilizadas na avaliação dos modelos de detecção estão:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

A avaliação considera as diferentes etapas do processo de estimativa de ocupação, incluindo:

1. detecção;
2. contagem de pessoas;
3. validação da contagem;
4. classificação do nível de ocupação.

A avaliação de contagem e classificação é tratada separadamente da avaliação de detecção.

---

## Requisitos

- Python 3.10+
- Ultralytics / YOLOv8
- PyTorch
- OpenCV
- NumPy
- Albumentations
- scikit-learn
- PyYAML
- ONNX / ONNX Runtime

As dependências utilizadas pelo projeto estão registradas em:

```text
requirements.txt
```

---

## Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
```

Entre na pasta:

```bash
cd person-training-pipeline
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Recomenda-se utilizar um ambiente virtual Python para a instalação das dependências.

---

## Organização dos experimentos

Os experimentos devem ser tratados de forma independente:

```text
Exp1
└── Baseline YOLOv8n

Exp2
└── Bus Occupancy
    ├── Dataset
    ├── Modelo
    └── Resultados

Exp3
└── Head Only
    ├── Dataset
    ├── Modelo
    └── Resultados
```

**Não utilizar resultados, pesos ou imagens de um experimento como se pertencessem a outro.**

Essa separação é importante para garantir a rastreabilidade dos experimentos e a correta interpretação dos resultados no TCC.

---

## Observação sobre os dados

Os datasets utilizados nos experimentos não são incluídos diretamente no repositório GitHub.

O `.gitignore` impede o versionamento de:

```text
.venv/
.idea/
__pycache__/
data/augmented/
data/datasets/
data/external_test/
runs/
```

Os arquivos permanecem disponíveis localmente para execução do projeto.

---

## Status

Projeto concluído, com os experimentos, modelos e resultados organizados de forma independente para documentação e análise no contexto do TCC.

### Experimentos

- [x] Exp1 — Baseline YOLOv8n
- [x] Exp2 — Bus Occupancy
- [x] Exp3 — Head Only
- [x] Avaliação das métricas de detecção
- [x] Contagem de pessoas
- [x] Validação da contagem
- [x] Classificação do nível de ocupação
- [x] Análise dos resultados

---

## Contexto acadêmico

Projeto desenvolvido no contexto do **Trabalho de Conclusão de Curso (TCC)** em Ciência da Computação, com foco em **visão computacional, detecção de objetos e estimativa de ocupação em transporte público**.