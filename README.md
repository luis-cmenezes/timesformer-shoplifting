# TimeSformer Shoplifting Detection

Projeto para fine-tuning do modelo **TimeSformer (Time-Space Transformer)** como classificador binário de **Shoplifting** (vs. Normal) usando vídeos de vigilância. O modelo utiliza arquitetura transformer para classificação de vídeos com atenção temporal e espacial.

---

## 1) Overview

### O que este repositório faz

- Processa datasets públicos de vigilância (DCSASS, MNNIT, Shoplifting 2.0) padronizando vídeos para **224×224 px** a **25 FPS**.
- Implementa **amostragem temporal estratificada** para extrair exatamente **8 frames** por vídeo (padrão TimeSformer).
- Treina (fine-tune) um TimeSformer pré-treinado no Kinetics-400 para **classificação binária** com balanceamento de classes.
- Utiliza **data augmentation** específica para vídeos: flip horizontal e color jitter consistentes entre frames.

### Estrutura de pastas

- `data/standarized/` → vídeos padronizados organizados por classe (Normal/Shoplifting)
- `src/` → código principal (dataset, treinamento, preprocessamento)
- `results/` → outputs de treinamento (checkpoints, logs, modelos finais)

### Dataset processado

- **549 vídeos** total: 314 Normal + 235 Shoplifting
- **Resolução:** 224×224 pixels (padronizada)
- **FPS:** 25 (padronizado)

---

## 2) Pré-processamento de Dados

O pipeline converte datasets heterogêneos em entradas consistentes para o TimeSformer.

### 2.1 Datasets públicos suportados

#### A) DCSASS Dataset
- **Fonte:** Datasets com múltiplas "situações", cada uma contendo clipes numerados
- **Estratégia:** Identifica **blocos contíguos** de clipes com o mesmo rótulo
- **Contexto:** Para eventos de shoplifting, adiciona clipes anterior/posterior quando disponível
- **Saída:** Concatena clipes por bloco e padroniza resolução/FPS

#### B) Shoplifting MNNIT & Dataset 2.0
- **Fonte:** Vídeos curtos individuais já rotulados
- **Estratégia:** Cada vídeo = um exemplo de treinamento
- **Processamento:** Padronização direta de resolução e FPS

### 2.2 Script de preprocessamento

Execute o script unificado para processar todos os datasets:

```bash
python src/process_and_standarize_data.py
```

**O script faz:**
1. **Padronização FFmpeg:** Redimensiona para 224×224 com aspect ratio preservado (padding quando necessário)
2. **FPS normalizado:** Converte todos os vídeos para 25 FPS
3. **Estrutura unificada:** Organiza em `data/standarized/Normal/` e `data/standarized/Shoplifting/`
4. **Manifest:** Gera `manifest.csv` com caminhos absolutos e labels

> **Nota:** Edite os caminhos no início do arquivo `process_and_standarize_data.py` conforme sua estrutura de dados.

---

## 3) Arquitetura do Modelo

### TimeSformer

- **TimeSformer:** Transformer puro com atenção espaço-temporal, mais eficiente para sequências longas
- **Entrada:** 8 frames de 224×224×3
- **Fine-tuning:** Classificação binária Shoplifting vs Normal

### Componentes principais

#### Dataset (`src/dataset.py`)
- **Amostragem temporal estratificada:** Divide o vídeo em N segmentos, amostra 1 frame por segmento
- **Augmentation:** Flip horizontal (50%) + Color Jitter (brightness/contrast/saturation/hue)
- **Suporte ao Decord:** Leitura eficiente de vídeos com tensores PyTorch nativos

#### Modelo (`src/model_utils.py`)
- **Estratégias de freeze:**
  - `unfreeze_head`: Congela backbone, treina apenas classificador
  - `unfreeze_all`: Fine-tuning completo
- **Configuração automática:** Adapta head para 2 classes com labels {Normal: 0, Shoplifting: 1}

---

## 4) Treinamento

### Dependências

Instale o ambiente:

```bash
# Com pip
pip install -e .

# Com uv (recomendado)
uv sync
```

**Dependências principais:**
- `transformers`: Hugging Face Transformers (TimeSformer)
- `torch` + `torchvision`: PyTorch
- `decord`: Leitura eficiente de vídeos
- `scikit-learn`: Splits estratificados e métricas
- `tensorboard`: Logging de treinamento

### Executar treinamento

**Fine-tuning completo:**

```bash
python src/train.py \
    --model-name facebook/timesformer-base-finetuned-k400 \
    --freeze-strategy unfreeze_all \
    --num-frames 8 \
    --output-dir results/runs \
```

**Apenas classificador:**

```bash
python src/train.py \
    --freeze-strategy unfreeze_head
```

### Parâmetros importantes

- `--model-name`: Checkpoint base (`k400` ou `ssv2`)
- `--freeze-strategy`: `unfreeze_head` | `unfreeze_all`
- `--num-frames`: Número de frames amostrados (padrão: 8)

### Configurações de treinamento

- **Split:** 70% treino, 15% validação, 15% teste (estratificado)
- **Batch size:** 48 (ajuste conforme GPU disponível)
- **Learning rate:** 1e-3
- **Épocas:** 70 com early stopping (patience=3)
- **Métrica principal:** AUC-ROC (balanceamento de classes)
- **Loss:** CrossEntropyLoss com class weights automáticos

### Balanceamento de classes

O sistema calcula automaticamente **class weights** baseado na distribuição:
- Weight Normal = #Shoplifting / #Normal 
- Weight Shoplifting = #Normal / #Shoplifting

---

## 5) Resultados e Monitoramento

### Estrutura de saída

```
results/
├── runs/<run-name>/
│   ├── checkpoints/          # Checkpoints por época
│   ├── final_model/         # Modelo final treinado
│   └── ...
└── logs/<run-name>/         # Logs TensorBoard
```

### Métricas acompanhadas

- **AUC-ROC**: Métrica principal para seleção de modelo
- **F1-Score**: Harmônica entre precisão e recall
- **Accuracy**: Acurácia global
- **Precision/Recall**: Métricas por classe

## Notas importantes

- **Labels automáticos:** Baseados no nome da pasta (Normal → 0, Shoplifting → 1)
- **Requerimento FFmpeg:** Necessário para preprocessamento de vídeos
- **GPU recomendada:** 8GB+ para batch_size=48, ajuste conforme disponibilidade
- **Reproducibilidade:** Seeds fixas para splits e amostragem temporal