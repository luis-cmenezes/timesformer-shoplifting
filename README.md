# TimeSformer Shoplifting Detection

Subpacote para fine-tuning do modelo **TimeSformer (Time-Space Transformer)** como classificador binário de **Shoplifting** (vs. Normal) usando vídeos de vigilância. O modelo utiliza arquitetura transformer com atenção temporal e espacial.

---

## 1) Overview

### O que este subpacote faz

- Fornece módulos de **preprocessamento** que padronizam datasets públicos de vigilância para **224×224 px** a **25 FPS** (vídeos `.mp4`).
- Implementa amostragem temporal estratificada para extrair **8 frames** por vídeo (padrão TimeSformer).
- Treina (fine-tune) um TimeSformer pré-treinado no Kinetics-400 para **classificação binária** com balanceamento de classes via `CrossEntropyLoss` com class weights.
- Utiliza data augmentation para vídeos: flip horizontal e color jitter consistentes entre frames.

### Estrutura do pacote

```
src/timesformer_shoplifting/
├── dataset/
│   └── dataset.py           # Dataset PyTorch com Decord + augmentation
├── inference/                # (reservado para uso futuro)
├── models/
│   └── model_utils.py        # Carregamento do TimeSformer + estratégias de freeze
├── preprocessing/
│   └── process_and_standardize_data.py  # Utilitários de padronização de vídeo (FFmpeg)
└── training/
    └── train.py              # Fine-tuning com HuggingFace Trainer
```

### Estrutura de dados

- `data/standarized/Normal/` → vídeos padronizados da classe Normal
- `data/standarized/Shoplifting/` → vídeos padronizados da classe Shoplifting
- `results/runs/<run-name>/` → checkpoints e modelo final
- `results/logs/<run-name>/` → logs TensorBoard

> **Nota:** quando executado via scripts centralizados (raiz do monorepo), os caminhos de dados são controlados pelo `scripts/config.yaml` (saída em `datasets/preprocessed/timesformer/standardized/`).

---

## 2) Preprocessamento

O pipeline converte datasets heterogêneos em vídeos padronizados para o TimeSformer.

> **Recomendado:** use o script centralizado `scripts/preprocess_timesformer.py` na raiz do monorepo, que orquestra todas as etapas de preprocessamento com caminhos configuráveis via `config.yaml`.

### Módulo de preprocessamento

`src/timesformer_shoplifting/preprocessing/process_and_standardize_data.py`

Este módulo é uma **biblioteca de funções** (sem CLI própria) usada pelos scripts centralizados. Fornece:

- `ensure_ffmpeg_exists()` — Verifica disponibilidade do FFmpeg
- `run_ffmpeg_concat_and_standardize()` — Concatena e padroniza vídeos (resolução + FPS)
- `load_annotations()` — Carrega anotações do DCSASS
- `identify_event_blocks_with_context()` — Identifica blocos de eventos com contexto temporal
- `find_videos_recursively()` — Busca vídeos em diretórios
- `process_simple_dataset()` — Processa datasets simples (MNNIT, Dataset 2.0)
- `generate_manifest()` — Gera `manifest.csv` com caminhos e labels

### Datasets suportados

| Dataset | Estratégia |
|---|---|
| **DCSASS** | Blocos contíguos de clipes com mesmo rótulo; contexto temporal para Shoplifting |
| **MNNIT** | Um vídeo = um exemplo; padronização direta |
| **Shoplifting Dataset 2.0** | Um vídeo = um exemplo; padronização direta |

### Padronização aplicada

- **Resolução:** 224×224 px (com padding para preservar aspect ratio)
- **FPS:** 25 (normalizado)
- **Formato:** `.mp4`

---

## 3) Modelo

### Arquitetura (`models/model_utils.py`)

- `get_model_and_processor(model_name, num_labels=2)` — Carrega `TimesformerForVideoClassification` do HuggingFace com `ignore_mismatched_sizes=True`, substituindo a cabeça Kinetics-400 por classificação binária (2 classes).
- `set_freeze_strategy(model, strategy)` — Estratégias de congelamento:
  - `unfreeze_head`: congela o backbone, treina apenas o classificador
  - `unfreeze_all`: fine-tuning completo de todas as camadas

### Dataset (`dataset/dataset.py`)

- `VideoAugmentation` — Horizontal flip (50%) + color jitter consistente entre frames.
- `SecurityVideoDataset` — Usa **Decord** para leitura eficiente de vídeos. `AutoImageProcessor` do HuggingFace faz resize/crop/normalização. Amostragem temporal uniforme (aleatória em treino, centro em validação). Falha suave: retorna tensor zerado se o arquivo estiver corrompido.

---

## 4) Treinamento

Script: `src/timesformer_shoplifting/training/train.py`

### O que o treino faz

- Carrega vídeos padronizados de `data/standarized/`
- Split estratificado: 70% treino, 15% validação, 15% teste
- `CrossEntropyLoss` com class weights automáticos para balanceamento
- Data augmentation: flip horizontal + color jitter
- HuggingFace `Trainer` com TensorBoard e Mixed Precision (fp16)
- Early stopping (paciência = 3 épocas) por AUC-ROC
- Salva melhor modelo em `<output-dir>/<run-name>/`

### Argumentos CLI

| Argumento | Default | Descrição |
|---|---|---|
| `--model-name` | `facebook/timesformer-base-finetuned-k400` | Checkpoint base (Kinetics-400 ou SSv2) |
| `--freeze-strategy` | `unfreeze_head` | `unfreeze_head` ou `unfreeze_all` |
| `--num-frames` | `8` | Número de frames amostrados por vídeo |
| `--output-dir` | `results/runs` | Diretório base para salvar execuções |
| `--log-dir` | `results/logs` | Diretório base para logs TensorBoard |
| `--data-root` | `data/standarized` | Diretório raiz dos dados padronizados |

### Parâmetros fixos (hardcoded no `TrainingArguments`)

| Parâmetro | Valor |
|---|---|
| Batch size | 48 |
| Learning rate | 1e-3 |
| Épocas | 70 |
| Early stopping patience | 3 |
| Precision | fp16 |

### Métricas acompanhadas

- **AUC-ROC** (métrica principal para seleção de modelo)
- **F1-Score**, **Accuracy**, **Precision**, **Recall**

### Exemplos

Fine-tuning completo:

```bash
uv run python src/timesformer_shoplifting/training/train.py \
    --model-name facebook/timesformer-base-finetuned-k400 \
    --freeze-strategy unfreeze_all \
    --num-frames 8 \
    --output-dir results/runs
```

Apenas classificador (head):

```bash
uv run python src/timesformer_shoplifting/training/train.py \
    --freeze-strategy unfreeze_head
```

### Dependências

```bash
uv sync
```

- **Runtime:** `torch`, `torchvision`, `transformers`, `accelerate`, `decord`, `evaluate`, `pandas`, `scikit-learn`, `tensorboard`, `tqdm`
- **Sistema:** `ffmpeg` (necessário para preprocessamento)

---

## Notas importantes

- **Labels automáticos:** baseados no nome da pasta (`Normal` → 0, `Shoplifting` → 1).
- **Preprocessamento é um módulo de biblioteca**, sem CLI própria. Use `scripts/preprocess_timesformer.py` na raiz do monorepo para executar o pipeline completo.
- **GPU recomendada:** 8GB+ para `batch_size=48`. Ajuste no código se necessário.
- **Reproducibilidade:** seeds fixas para splits e amostragem temporal.
