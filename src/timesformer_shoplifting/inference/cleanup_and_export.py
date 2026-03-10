"""
Para cada experimento em results/timesformer/:
  1. Gera tb_metrics.csv em final_model/ com TODAS as métricas (treino + eval) dos tfevents
  2. Deleta trainer_state.json (parcial/redundante — tfevents tem tudo)
  3. Deleta train_results.json e all_results.json (duplicados — contidos no CSV)

Uso:
  uv run cleanup_and_export.py            # dry-run (mostra o que faria)
  uv run cleanup_and_export.py --execute  # executa de verdade
"""

import argparse
from pathlib import Path

import pandas as pd
from tbparse import SummaryReader


RESULTS_DIR = Path("results/timesformer")


def process_experiment(exp: Path, *, dry_run: bool) -> None:
    runs_dir = exp / "checkpoints" / "runs"
    final_dir = exp / "final_model"
    name = exp.name

    if not runs_dir.exists():
        print(f"  SKIP {name}: sem checkpoints/runs/")
        return

    # 1. Gerar CSV com todas as métricas do TensorBoard
    csv_path = final_dir / "tb_metrics.csv"
    reader = SummaryReader(str(runs_dir))
    df = reader.scalars

    if df.empty:
        print(f"  SKIP {name}: tfevents vazios")
        return

    # Pivotar para formato wide: uma linha por step, uma coluna por métrica
    wide = df.pivot_table(index="step", columns="tag", values="value", aggfunc="first")
    wide = wide.sort_index().reset_index()
    wide.columns.name = None  # remove nome do MultiIndex

    if dry_run:
        print(f"  CSV  {name}: {len(wide)} rows × {len(wide.columns)} cols → {list(wide.columns)}")
    else:
        wide.to_csv(csv_path, index=False)
        print(f"  CSV  {name}: {len(wide)} rows × {len(wide.columns)} cols → {csv_path}")

    # 2. Deletar arquivos redundantes
    to_delete = [
        final_dir / "trainer_state.json",
        exp / "checkpoints" / "train_results.json",
        exp / "checkpoints" / "all_results.json",
    ]
    for f in to_delete:
        if f.exists():
            if dry_run:
                print(f"  DEL  {f}")
            else:
                f.unlink()
                print(f"  DEL  {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Executa (default: dry-run)")
    args = parser.parse_args()

    dry_run = not args.execute
    if dry_run:
        print("=== DRY RUN (use --execute para aplicar) ===\n")
    else:
        print("=== EXECUTANDO ===\n")

    for exp in sorted(RESULTS_DIR.iterdir()):
        if exp.is_dir():
            process_experiment(exp, dry_run=dry_run)
            print()

    if dry_run:
        print("Nada foi alterado. Rode com --execute para aplicar.")


if __name__ == "__main__":
    main()
