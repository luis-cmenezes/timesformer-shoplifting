import os
import argparse
import re
import torch
import torch.nn as nn
import numpy as np
import evaluate
from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from timesformer_shoplifting.dataset.dataset import SecurityVideoDataset
from timesformer_shoplifting.models.model_utils import get_model_and_processor, set_freeze_strategy

# Configuração de Logs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silencia logs verbosos de requisições HTTP do HuggingFace Hub / httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Dataclass de configuração - interface pública para chamadas externas
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Configuração completa para um experimento de treino TimeSformer."""

    # Checkpoint base do HuggingFace
    model_name: str = "facebook/timesformer-base-finetuned-k400"

    # Estratégia de fine-tuning: "unfreeze_head" | "unfreeze_all"
    freeze_strategy: str = "unfreeze_head"

    # Número de frames amostrados por vídeo
    num_frames: int = 8

    # Hiperparâmetros
    epochs: int = 70
    batch_size: int = 48
    learning_rate: float = 1e-3
    seed: int = 42
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 5
    logging_steps: int = 10

    # Caminhos
    data_root: str = ""
    output_dir: str = ""
    log_dir: str = ""

    # Divisão do dataset
    split_test_size: float = 0.3
    split_val_test_ratio: float = 0.5

    # Data augmentation
    augmentation_p_flip: float = 0.5
    augmentation_color_jitter: dict = field(
        default_factory=lambda: dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    )

    # Early stopping
    early_stopping_patience: int = 3

def compute_metrics(eval_pred):
    """
    Calcula métricas robustas para classificação, incluindo AUC-ROC.
    """
    predictions, labels = eval_pred
    # Timesformer retorna logits (batch_size, 2), pegamos as probabilidades da classe positiva
    # softmax([logit0, logit1])[1] = exp(logit1) / (exp(logit0) + exp(logit1))
    probs_positive = np.exp(predictions[:, 1]) / (np.exp(predictions[:, 0]) + np.exp(predictions[:, 1]))
    predictions_binary = np.argmax(predictions, axis=1)
    
    # Carregar métricas do HF Evaluate
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    acc_res = accuracy.compute(predictions=predictions_binary, references=labels)
    # average="binary" pois temos apenas 2 classes
    f1_res = f1.compute(predictions=predictions_binary, references=labels, average="binary", pos_label=1)
    prec_res = precision.compute(predictions=predictions_binary, references=labels, average="binary", pos_label=1)
    rec_res = recall.compute(predictions=predictions_binary, references=labels, average="binary", pos_label=1)
    
    # Calcular AUC-ROC (métrica principal para seleção de modelo, como no I3D)
    auc_roc = roc_auc_score(labels, probs_positive)
    
    return {
        "accuracy": acc_res["accuracy"],
        "f1": f1_res["f1"],
        "precision": prec_res["precision"],
        "recall": rec_res["recall"],
        "auc_roc": auc_roc  # Nova métrica para seleção de modelo
    }

def collate_fn(batch):
    """
    Processa a lista de amostras em um batch de tensores empilhados.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


class CustomTrainerWithClassWeights(Trainer):
    """
    Trainer customizado que aplica class weights na função de perda.
    Implementa balanceamento de classes como no I3D.
    """
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Calcula a loss com class weights.
        
        Args:
            model: Modelo de classificação
            inputs: Batch de entrada com 'labels'
            return_outputs: Se True, retorna também os outputs do modelo
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Aplicar CrossEntropyLoss com class weights
        if self.class_weights is not None:
            class_weights_device = self.class_weights.to(model.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights_device)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def parse_args() -> TrainConfig:
    """Analisa argumentos CLI e devolve um TrainConfig (uso standalone)."""
    parser = argparse.ArgumentParser(description="Treino TimeSformer para Shoplifting")
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/timesformer-base-finetuned-k400",
        choices=[
            "facebook/timesformer-base-finetuned-k400",
            "facebook/timesformer-base-finetuned-ssv2",
        ],
        help="Checkpoint base do TimeSformer (Kinetics-400 ou SSv2)",
    )
    parser.add_argument(
        "--freeze-strategy",
        type=str,
        default="unfreeze_head",
        choices=["unfreeze_head", "unfreeze_all"],
        help="unfreeze_head congela o backbone e treina apenas o classificador (head); unfreeze_all faz fine-tuning completo",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Número de frames amostrados por vídeo (aumentar para aproximar o I3D)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/runs",
        help="Diretório base para salvar execuções (cada run cria uma subpasta com o método de treino)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/logs",
        help="Diretório base para logs do TensorBoard (cada run cria uma subpasta)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/standarized",
        help="Diretório raiz dos dados padronizados",
    )

    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        freeze_strategy=args.freeze_strategy,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        data_root=args.data_root,
    )


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("/", "-")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "run"


def build_run_name(model_name: str, num_frames: int, freeze_strategy: str) -> str:
    model_id = model_name.split("/")[-1]
    return _slugify(f"{model_id}_frames{num_frames}_{freeze_strategy}")


def train(cfg: TrainConfig):
    """Função principal que orquestra o treinamento do TimeSformer.

    Args:
        cfg: instância de TrainConfig com todos os parâmetros do experimento.
    """
    # Definições de Diretórios
    DATA_ROOT = os.path.abspath(cfg.data_root)

    run_name = build_run_name(
        model_name=cfg.model_name,
        num_frames=cfg.num_frames,
        freeze_strategy=cfg.freeze_strategy,
    )

    # Estrutura de saída:
    # <output-dir>/<run-name>/checkpoints
    # <log-dir>/<run-name>
    # <output-dir>/<run-name>/final_model
    RUN_ROOT = os.path.join(cfg.output_dir, run_name)
    OUTPUT_DIR = os.path.join(RUN_ROOT, "checkpoints")
    LOG_DIR = os.path.join(cfg.log_dir, run_name)
    FINAL_MODEL_DIR = os.path.join(RUN_ROOT, "final_model")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    print(f"Usando dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Modelo: {cfg.model_name}")
    print(f"Estratégia: {cfg.freeze_strategy} | Frames: {cfg.num_frames}")
    print(f"Outputs: {RUN_ROOT}")
    
    # 1. Carregar Modelo e Processador
    model, processor = get_model_and_processor(
        cfg.model_name, num_labels=2, num_frames=cfg.num_frames,
    )
    set_freeze_strategy(model, strategy=cfg.freeze_strategy)

    # Preparar parâmetros de augmentação
    aug_color_jitter = cfg.augmentation_color_jitter if cfg.augmentation_color_jitter else None
    
    # 2. Preparar Dataset
    # Instanciamos um dataset temporário para obter os índices e labels
    full_dataset_temp = SecurityVideoDataset(
        DATA_ROOT, processor, num_frames=cfg.num_frames, split="train",
        augmentation_p_flip=cfg.augmentation_p_flip,
        augmentation_color_jitter=aug_color_jitter,
    )
    all_indices = list(range(len(full_dataset_temp)))
    all_labels = full_dataset_temp.labels
    
    # Split estratificado configurável
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=cfg.split_test_size,
        stratify=all_labels,
        random_state=cfg.seed,
    )
    
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=cfg.split_val_test_ratio,
        stratify=temp_labels,
        random_state=cfg.seed,
    )
    
    # Criar datasets com split apropriado para cada um
    train_dataset = torch.utils.data.Subset(
        SecurityVideoDataset(
            DATA_ROOT, processor, num_frames=cfg.num_frames, split="train",
            augmentation_p_flip=cfg.augmentation_p_flip,
            augmentation_color_jitter=aug_color_jitter,
        ),
        train_idx,
    )
    
    val_dataset = torch.utils.data.Subset(
        SecurityVideoDataset(
            DATA_ROOT, processor, num_frames=cfg.num_frames, split="val",
            augmentation_p_flip=cfg.augmentation_p_flip,
            augmentation_color_jitter=aug_color_jitter,
        ),
        val_idx,
    )
    
    # Imprimir estatísticas do split
    print("\n=== Dataset Split ===")
    print(f"Dataset Treino:     {len(train_dataset)} vídeos ({100*len(train_idx)/len(all_indices):.1f}%)")
    print(f"Dataset Validação:  {len(val_dataset)} vídeos ({100*len(val_idx)/len(all_indices):.1f}%)")
    
    # Verificar stratificação
    train_labels = [all_labels[i] for i in train_idx]
    val_labels = [all_labels[i] for i in val_idx]
    
    print("\n=== Distribuição de Classes ===")
    print(f"Treino  - Normal: {sum(1 for label in train_labels if label == 0)}, Shoplifting: {sum(1 for label in train_labels if label == 1)}")
    print(f"Val     - Normal: {sum(1 for label in val_labels if label == 0)}, Shoplifting: {sum(1 for label in val_labels if label == 1)}\n")
    
    # Calcular class weights para balanceamento
    num_normal = sum(1 for label in train_labels if label == 0)
    num_shoplifting = sum(1 for label in train_labels if label == 1)
    
    weight_normal = num_shoplifting / num_normal
    weight_shoplifting = num_normal / num_shoplifting
    
    class_weights = torch.tensor([weight_normal, weight_shoplifting], dtype=torch.float32)
    
    print("=== Class Weights (Balanceamento) ===")
    print(f"Peso da classe Normal (0):      {weight_normal:.4f}")
    print(f"Peso da classe Shoplifting (1): {weight_shoplifting:.4f}\n")
    
    # 3. Argumentos de Treinamento (Hyperparameters)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.epochs,
        seed=cfg.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc_roc",
        logging_dir=LOG_DIR,
        logging_steps=cfg.logging_steps,
        dataloader_num_workers=cfg.dataloader_num_workers,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="tensorboard",
        greater_is_better=True,
    )
    
    # 4. Inicializar Trainer
    early_stopping_patience = cfg.early_stopping_patience
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience > 0 else []

    trainer = CustomTrainerWithClassWeights(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        class_weights=class_weights,
        callbacks=callbacks,
    )
    
    # 5. Iniciar Treino
    print("Iniciando treinamento...")
    train_result = trainer.train()
    
    # 6. Salvar Modelo Final
    print(f"Salvando modelo em {FINAL_MODEL_DIR}")
    trainer.save_model(FINAL_MODEL_DIR)
    
    # Log de métricas finais
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Avaliação Final em Validação
    print("\n=== Avaliação em VALIDAÇÃO ===")
    val_metrics = trainer.evaluate()
    trainer.log_metrics("eval", val_metrics)
    
    # Resumo Final
    print("\n" + "="*50)
    print("RESUMO DE MÉTRICAS")
    print("="*50)
    print(f"Validação - F1: {val_metrics.get('eval_f1', 'N/A'):.4f}, AUC: {val_metrics.get('eval_auc_roc', 'N/A'):.4f}")
    print("="*50 + "\n")
    
if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)