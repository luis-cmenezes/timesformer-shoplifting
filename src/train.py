import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import evaluate
from sklearn.metrics import roc_auc_score
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from dataset import SecurityVideoDataset
from model_utils import get_model_and_processor, set_freeze_strategy

# Configuração de Logs
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def parse_args():
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
        default="results/checkpoints",
        help="Diretório para salvar checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/logs",
        help="Diretório para logs do TensorBoard",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/standarized",
        help="Diretório raiz dos dados padronizados",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Definições de Diretórios
    DATA_ROOT = os.path.abspath(args.data_root)
    OUTPUT_DIR = args.output_dir
    LOG_DIR = args.log_dir
    
    # 1. Carregar Modelo e Processador
    model, processor = get_model_and_processor(args.model_name, num_labels=2)
    set_freeze_strategy(model, strategy=args.freeze_strategy)
    
    # 2. Preparar Dataset
    # Instanciamos um dataset temporário para obter os índices e labels
    full_dataset_temp = SecurityVideoDataset(DATA_ROOT, processor, num_frames=args.num_frames, split="train")
    all_indices = list(range(len(full_dataset_temp)))
    all_labels = full_dataset_temp.labels
    
    # Split 70/15/15 com estratificação
    # Primeira divisão: 70% treino, 30% para validação + teste
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.3,
        stratify=all_labels,
        random_state=42
    )
    
    # Segunda divisão: divide os 30% em 50/50 → 15% validação, 15% teste
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )
    
    # Criar datasets com split apropriado para cada um
    train_dataset = torch.utils.data.Subset(
        SecurityVideoDataset(DATA_ROOT, processor, num_frames=args.num_frames, split="train"),
        train_idx
    )
    
    val_dataset = torch.utils.data.Subset(
        SecurityVideoDataset(DATA_ROOT, processor, num_frames=args.num_frames, split="val"),
        val_idx
    )
    
    # Imprimir estatísticas do split
    print("\n=== Dataset Split (70/30 em dois estágios: train/val) ===")
    print(f"Dataset Treino:     {len(train_dataset)} vídeos ({100*len(train_idx)/len(all_indices):.1f}%)")
    print(f"Dataset Validação:  {len(val_dataset)} vídeos ({100*len(val_idx)/len(all_indices):.1f}%)")
    
    # Verificar stratificação
    train_labels = [all_labels[i] for i in train_idx]
    val_labels = [all_labels[i] for i in val_idx]
    
    print("\n=== Distribuição de Classes ===")
    print(f"Treino  - Normal: {sum(1 for label in train_labels if label == 0)}, Shoplifting: {sum(1 for label in train_labels if label == 1)}")
    print(f"Val     - Normal: {sum(1 for label in val_labels if label == 0)}, Shoplifting: {sum(1 for label in val_labels if label == 1)}\n")
    
    # Calcular class weights para balanceamento (como no I3D)
    num_normal = sum(1 for label in train_labels if label == 0)
    num_shoplifting = sum(1 for label in train_labels if label == 1)
    
    # Fórmula: peso = num_classe_majoritária / num_classe_minoritária
    # Se classe é minoritária, recebe peso > 1
    weight_normal = num_shoplifting / num_normal
    weight_shoplifting = num_normal / num_shoplifting
    
    class_weights = torch.tensor([weight_normal, weight_shoplifting], dtype=torch.float32)
    
    print("=== Class Weights (Balanceamento) ===")
    print(f"Peso da classe Normal (0):      {weight_normal:.4f}")
    print(f"Peso da classe Shoplifting (1): {weight_shoplifting:.4f}\n")
    
    # 3. Argumentos de Treinamento (Hyperparameters)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-3,              # LR baixo para Fine-tuning
        per_device_train_batch_size=48, 
        per_device_eval_batch_size=48,
        gradient_accumulation_steps=1,
        num_train_epochs=70,             
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc_roc",  # AUC-ROC em vez de F1 (como no I3D)
        logging_dir=LOG_DIR,
        logging_steps=10,
        dataloader_num_workers=5,        # Paralelismo no carregamento do Decord
        fp16=torch.cuda.is_available(),  # Mixed Precision (Nvidia GPUs only)
        remove_unused_columns=False,     # Importante para datasets customizados
        report_to="tensorboard",
        greater_is_better=True
    )
    
    # 4. Inicializar Trainer
    trainer = CustomTrainerWithClassWeights(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,             # Passamos o processor como tokenizer para salvar config
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        class_weights=class_weights,     # Passa os pesos das classes
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Para se não melhorar em 3 épocas
    )
    
    # 5. Iniciar Treino
    print("Iniciando treinamento...")
    train_result = trainer.train()
    
    # 6. Salvar Modelo Final
    print("Salvando modelo em results/final_model")
    trainer.save_model("results/final_model")
    
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
    main()