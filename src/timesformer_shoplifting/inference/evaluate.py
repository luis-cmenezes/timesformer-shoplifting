"""
Script de inferência e avaliação no conjunto de TESTE para o modelo TimeSformer.

===========================================================================
PASSO A PASSO — o que este script faz e por quê:
===========================================================================

1. CARREGAR O MELHOR MODELO DO TREINAMENTO
   ─ O Hugging Face Trainer com ``load_best_model_at_end=True`` salva o
     melhor modelo (por AUC-ROC) em ``final_model/``.  Utilizamos
     ``TimesformerForVideoClassification.from_pretrained(final_model_dir)``
     para recarregá-lo, junto com o ``AutoImageProcessor`` salvo ao
     lado (``preprocessor_config.json``).
   ─ Alternativamente, se ``final_model/`` não existir, o script busca
     o melhor checkpoint em ``checkpoints/`` via ``trainer_state.json``.

2. NÃO É NECESSÁRIO CALIBRAR THRESHOLD
   ─ Diferente do I3D (que usa ``BCEWithLogitsLoss`` com saída escalar),
     o TimeSformer usa ``CrossEntropyLoss`` com 2 logits (Normal,
     Shoplifting).  A predição é simplesmente ``argmax`` dos logits.
   ─ Mesmo assim, calculamos as probabilidades via softmax para gerar
     a curva ROC e AUC-ROC (usando a probabilidade da classe 1).

3. RECRIAR O SPLIT DE TESTE COM A MESMA SEED
   ─ Reproduzimos exatamente o mesmo ``train_test_split`` estratificado
     do treino, com mesma seed, mesmos parâmetros e mesma ordenação
     do dataset.  Isso garante que o conjunto de teste *nunca* foi
     visto durante o treinamento.

4. INFERÊNCIA E CONFRONTAMENTO COM GROUND TRUTH
   ─ Iteramos sobre o conjunto de teste com o modelo em ``eval()`` mode.
   ─ Coletamos logits, probabilidades (softmax) e labels verdadeiros.

5. GERAÇÃO DE MÉTRICAS COMPLETAS
   ─ Accuracy, Precision, Recall, F1, AUC-ROC
   ─ Matriz de Confusão, Classification Report
   ─ Curva ROC (PNG + dados brutos .npz)
   ─ Tudo é salvo em ``test_evaluation/`` dentro do diretório do
     experimento.

===========================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoImageProcessor, TimesformerForVideoClassification

from timesformer_shoplifting.dataset.dataset import SecurityVideoDataset


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EvalConfig:
    """Parâmetros para a avaliação de um experimento TimeSformer."""

    # Diretório raiz do experimento (contém checkpoints/ e final_model/)
    experiment_dir: str

    # Diretório raiz dos dados padronizados (Normal/ e Shoplifting/)
    data_root: str

    # Número de frames amostrados (deve ser o mesmo do treino)
    num_frames: int = 8

    # Seed e split idênticos ao treino
    seed: int = 42
    split_test_size: float = 0.3
    split_val_test_ratio: float = 0.5

    # Batch size para inferência
    batch_size: int = 16

    # Número de workers para DataLoader
    dataloader_num_workers: int = 4


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def find_best_model_dir(experiment_dir: Path) -> Path:
    """Encontra o diretório do melhor modelo salvo pelo Trainer.

    Prioridade:
      1. ``final_model/`` — salvo por ``trainer.save_model()`` ao final
         do treino com ``load_best_model_at_end=True``.
      2. O checkpoint indicado em ``trainer_state.json`` como
         ``best_model_checkpoint``.

    Returns:
        Path para o diretório contendo ``model.safetensors`` / ``config.json``.
    """
    # Opção 1: final_model/
    final_model = experiment_dir / "final_model"
    if (final_model / "config.json").exists():
        print(f"Usando modelo final: {final_model}")
        return final_model

    # Opção 2: melhor checkpoint de trainer_state.json
    checkpoints_dir = experiment_dir / "checkpoints"
    for ckpt_dir in sorted(checkpoints_dir.iterdir()):
        state_file = ckpt_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            best_ckpt = state.get("best_model_checkpoint")
            if best_ckpt and Path(best_ckpt).exists():
                print(f"Usando melhor checkpoint: {best_ckpt}")
                return Path(best_ckpt)

    raise FileNotFoundError(
        f"Nenhum modelo encontrado em {experiment_dir}. "
        "Esperado 'final_model/' ou um 'trainer_state.json' com 'best_model_checkpoint'."
    )


def load_model_and_processor(model_dir: Path):
    """Carrega o modelo TimeSformer e o processador de imagem.

    Returns:
        (model, processor) — ambos prontos para inferência.
    """
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = TimesformerForVideoClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    print(f"Modelo carregado com {model.config.num_labels} labels: "
          f"{model.config.id2label}")
    return model, processor


def reproduce_splits(dataset, seed: int, test_size: float, val_test_ratio: float):
    """Reproduz exatamente os splits de treino/validação/teste.

    O treino faz a divisão usando ``all_indices`` e ``all_labels`` do
    ``SecurityVideoDataset`` (que usa ``_build_index`` para listar
    vídeos por classe).  Recreamos a mesma sequência aqui.

    Returns:
        (train_idx, val_idx, test_idx)
    """
    all_indices = list(range(len(dataset)))
    all_labels = dataset.labels

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=test_size,
        stratify=all_labels,
        random_state=seed,
    )

    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=val_test_ratio,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def collate_fn(batch):
    """Mesmo collate_fn usado no treino."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executa inferência no TimeSformer.

    O modelo retorna logits de shape (B, 2).  Aplicamos softmax para
    obter probabilidades e argmax para predições binárias.

    Returns:
        (probs_positive, predictions, labels) — arrays numpy.
        ``probs_positive`` contém P(Shoplifting) para cada amostra.
    """
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(dataloader, desc="Inferência"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, 2)

        # Softmax para probabilidades
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_pos = probs[:, 1]  # P(Shoplifting)
        preds = np.argmax(probs, axis=1)

        all_probs.extend(probs_pos.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Métricas e relatórios
# ---------------------------------------------------------------------------

def compute_and_save_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    output_dir: Path,
):
    """Calcula todas as métricas e salva artefatos no ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Métricas escalares ---
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    auc_roc = roc_auc_score(labels, probs)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_roc,
        "num_samples": int(len(labels)),
        "num_positive": int(labels.sum()),
        "num_negative": int((labels == 0).sum()),
    }

    # Salva métricas como JSON
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétricas salvas em {metrics_path}")

    # --- Classification Report ---
    report_str = classification_report(
        labels, preds,
        target_names=["Normal", "Shoplifting"],
        digits=4,
    )
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report_str)

    # --- Matriz de Confusão ---
    cm = confusion_matrix(labels, preds)
    cm_path = output_dir / "confusion_matrix.txt"
    cm_path.write_text(
        f"Confusion Matrix:\n"
        f"               Pred Normal  Pred Shoplifting\n"
        f"True Normal     {cm[0, 0]:>10}  {cm[0, 1]:>16}\n"
        f"True Shoplift   {cm[1, 0]:>10}  {cm[1, 1]:>16}\n"
    )

    # --- Curva ROC (dados brutos) ---
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    roc_auc_val = auc(fpr, tpr)
    np.savez(
        output_dir / "roc_curve_test.npz",
        fpr=fpr, tpr=tpr, thresholds=thresholds_roc, auc=roc_auc_val,
    )

    # --- Curva ROC (imagem PNG) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Aleatório")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC — Conjunto de Teste (TimeSformer)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve_test.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib não disponível — pulando geração do PNG da curva ROC.")

    # --- Print resumo ---
    print("\n" + "=" * 60)
    print("RESULTADOS NO CONJUNTO DE TESTE (TimeSformer)")
    print("=" * 60)
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc_roc:.4f}")
    print(f"  Amostras:   {len(labels)} (Pos={int(labels.sum())}, Neg={int((labels==0).sum())})")
    print("=" * 60)
    print(f"\nRelatório completo:\n{report_str}")
    print(f"Matriz de Confusão:\n{cm}\n")

    return metrics


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def evaluate(cfg: EvalConfig) -> dict:
    """Executa o pipeline completo de avaliação no conjunto de teste.

    Args:
        cfg: instância de EvalConfig com todos os parâmetros.

    Returns:
        Dicionário com todas as métricas calculadas.
    """
    experiment_path = Path(cfg.experiment_dir)
    eval_output_dir = experiment_path / "test_evaluation"

    print(f"Dispositivo: {DEVICE}")
    print(f"Experimento: {experiment_path}")

    # =====================================================================
    # PASSO 1 — Carregar o melhor modelo
    # =====================================================================
    best_model_dir = find_best_model_dir(experiment_path)
    model, processor = load_model_and_processor(best_model_dir)

    # =====================================================================
    # PASSO 2 — Recriar o dataset (sem augmentation / modo "val")
    # =====================================================================
    # Usamos split="val" para desativar augmentation e usar temporal
    # sampling determinístico (ponto médio de cada segmento).
    full_dataset = SecurityVideoDataset(
        root_dir=cfg.data_root,
        image_processor=processor,
        num_frames=cfg.num_frames,
        split="val",  # sem augmentation, temporal sampling determinístico
    )
    print(f"Dataset total: {len(full_dataset)} vídeos")

    # =====================================================================
    # PASSO 3 — Reproduzir splits (mesma seed do treino)
    # =====================================================================
    train_idx, val_idx, test_idx = reproduce_splits(
        full_dataset,
        seed=cfg.seed,
        test_size=cfg.split_test_size,
        val_test_ratio=cfg.split_val_test_ratio,
    )

    print(f"Split — Treino: {len(train_idx)}, Validação: {len(val_idx)}, Teste: {len(test_idx)}")

    # Verificar distribuição no teste
    test_labels_preview = [full_dataset.labels[i] for i in test_idx]
    print(f"Teste — Normal: {sum(1 for lb in test_labels_preview if lb == 0)}, "
          f"Shoplifting: {sum(1 for lb in test_labels_preview if lb == 1)}")

    # =====================================================================
    # PASSO 4 — Inferência no conjunto de TESTE
    # =====================================================================
    print("\n--- Inferência no conjunto de TESTE ---")
    test_dataset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.dataloader_num_workers,
    )

    test_probs, test_preds, test_labels = run_inference(model, test_loader)

    # =====================================================================
    # PASSO 5 — Calcular e salvar métricas
    # =====================================================================
    metrics = compute_and_save_metrics(
        labels=test_labels,
        probs=test_probs,
        preds=test_preds,
        output_dir=eval_output_dir,
    )

    # Salvar probabilidades brutas para análises futuras
    np.savez(
        eval_output_dir / "test_predictions.npz",
        probs=test_probs,
        preds=test_preds,
        labels=test_labels,
        test_indices=np.array(test_idx),
    )
    print(f"Predições brutas salvas em {eval_output_dir / 'test_predictions.npz'}")

    return metrics


# ---------------------------------------------------------------------------
# Entrypoint standalone (uso direto via CLI)
# ---------------------------------------------------------------------------

def _parse_args() -> EvalConfig:
    import argparse

    parser = argparse.ArgumentParser(
        description="Avaliação TimeSformer no conjunto de teste",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Diretório do experimento (ex: results/timesformer/timesformer-base-...)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Diretório raiz dos dados padronizados",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-test-size", type=float, default=0.3)
    parser.add_argument("--split-val-test-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)

    args = parser.parse_args()
    return EvalConfig(
        experiment_dir=args.experiment_dir,
        data_root=args.data_root,
        num_frames=args.num_frames,
        seed=args.seed,
        split_test_size=args.split_test_size,
        split_val_test_ratio=args.split_val_test_ratio,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    evaluate(cfg)
