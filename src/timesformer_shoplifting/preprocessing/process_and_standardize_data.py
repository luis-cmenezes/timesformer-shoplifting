"""
Funções utilitárias de pré-processamento para o TimeSformer.

Responsabilidades:
  - Verificação de ffmpeg
  - Concatenação + padronização de vídeo (fps, resolução) via ffmpeg
  - Identificação de blocos de eventos do DCSASS (com contexto)
  - Processamento simplificado de datasets com vídeos individuais (MNNIT, Dataset 2.0)
  - Geração de manifest.csv
"""

import os
import csv
import subprocess

import pandas as pd
from tqdm import tqdm

# ── Constantes ───────────────────────────────────────────────────────────────

TMP_LIST_FILENAME = "tmp_ffmpeg_file_list.txt"
MANIFEST_FILENAME = "manifest.csv"


# ── Utilidades ───────────────────────────────────────────────────────────────

def ensure_ffmpeg_exists():
    """Verifica se o ffmpeg está disponível no PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except Exception as e:
        raise RuntimeError(
            "ffmpeg não encontrado no PATH. Instale o ffmpeg e tente novamente."
        ) from e


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


# ── FFmpeg: padronização de vídeo ────────────────────────────────────────────

def run_ffmpeg_concat_and_standardize(
    input_file_list, output_path, fps=25, w=224, h=224
):
    """
    Concatena a lista de arquivos (concat demuxer), padroniza FPS e resolução
    com scale+pad, e escreve em *output_path* (mp4).

    Args:
        input_file_list: caminho para arquivo de texto com linhas
            ``file '/abs/path/to/file'``
        output_path: caminho completo do vídeo final
        fps: taxa de frames de saída
        w, h: largura e altura alvo (pixels)

    Returns:
        tuple[bool, str | None]: (sucesso, mensagem_de_erro)
    """
    ratio = float(w) / float(h)

    scale_part = (
        f"scale='if(gt(iw/ih,{ratio}),{w},-2)':'if(gt(iw/ih,{ratio}),-2,{h})'"
    )
    pad_part = f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    vf = f"fps={fps},{scale_part},{pad_part}"

    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-y",
        "-i", str(input_file_list),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, None
    except subprocess.CalledProcessError as e:
        print(f"Erro ao processar {output_path}")
        print("---- FFmpeg stderr ----")
        print(e.stderr)
        print("-----------------------")
        return False, e.stderr


def write_ffmpeg_file_list(clip_paths, file_path):
    """Escreve arquivo no formato esperado pelo concat demuxer do FFmpeg."""
    with open(file_path, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")


# ── DCSASS: identificação de blocos com contexto ─────────────────────────────

def load_annotations(annotations_path):
    """Carrega anotações DCSASS em um DataFrame indexado pelo nome do clipe."""
    try:
        df = pd.read_csv(
            annotations_path,
            header=None,
            names=["clip_name_base", "category", "label"],
        )
        df.set_index("clip_name_base", inplace=True)
        print(f"Anotações carregadas de: {annotations_path}")
        return df
    except FileNotFoundError:
        print(f"Arquivo de anotações não encontrado em {annotations_path}")
        return None


def identify_event_blocks_with_context(dataset_root, annotations_df):
    """
    Identifica blocos de clipes contíguos com o mesmo rótulo e adiciona
    clipes de contexto (anterior/posterior) aos blocos de Shoplifting.

    Returns:
        list[dict]: cada dict tem ``'label'`` (int) e ``'clip_paths'`` (list[str]).
    """
    print("Identificando blocos de eventos e adicionando contexto...")
    all_blocks = []
    situation_folders = sorted(
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )

    for situation_name in tqdm(situation_folders, desc="Analisando Situações"):
        situation_path = os.path.join(dataset_root, situation_name)

        situation_clip_data = []
        try:
            sorted_clip_names = sorted(
                [f for f in os.listdir(situation_path) if f.endswith(".mp4")],
                key=lambda name: int(os.path.splitext(name)[0].split("_")[-1]),
            )
            for clip_filename in sorted_clip_names:
                clip_name_base = os.path.splitext(clip_filename)[0]
                label = annotations_df.loc[clip_name_base, "label"]
                situation_clip_data.append(
                    {"path": os.path.join(situation_path, clip_filename), "label": label}
                )
        except (ValueError, IndexError, KeyError) as e:
            print(
                f"\nAviso: Problema ao processar clipes em '{situation_name}': {e}. Pulando."
            )
            continue

        if not situation_clip_data:
            continue

        # Identifica blocos contíguos
        i = 0
        while i < len(situation_clip_data):
            start_index = i
            current_label = situation_clip_data[start_index]["label"]

            end_index = start_index
            while (
                end_index + 1 < len(situation_clip_data)
                and situation_clip_data[end_index + 1]["label"] == current_label
            ):
                end_index += 1

            block_clip_paths = [
                d["path"] for d in situation_clip_data[start_index : end_index + 1]
            ]

            # Contexto para blocos de Shoplifting
            if current_label == 1:
                if start_index > 0:
                    block_clip_paths.insert(
                        0, situation_clip_data[start_index - 1]["path"]
                    )
                if end_index < len(situation_clip_data) - 1:
                    block_clip_paths.append(
                        situation_clip_data[end_index + 1]["path"]
                    )

            all_blocks.append({"label": current_label, "clip_paths": block_clip_paths})
            i = end_index + 1

    print(
        f"Identificação concluída. Total de {len(all_blocks)} blocos de eventos encontrados."
    )
    return all_blocks


# ── Processamento simples (MNNIT / Dataset 2.0) ─────────────────────────────

def find_videos_recursively(root_dir):
    """Gerador que retorna caminhos absolutos de vídeos encontrados recursivamente."""
    exts = (".mp4", ".avi", ".mov", ".mkv")
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)


def process_simple_dataset(
    source_root, mapping_folder_to_label, dest_root, prefix, fps=25, w=224, h=224
):
    """
    Processa vídeos simples (MNNIT / Dataset 2.0): padroniza FPS e resolução,
    salva em pasta de saída e retorna lista de entradas para o manifest.

    Args:
        source_root: pasta raiz do dataset
        mapping_folder_to_label: dict de pasta -> label (0/1)
        dest_root: pasta raiz de saída
        prefix: string para prefixar os arquivos
        fps: fps alvo
        w, h: resolução alvo

    Returns:
        list[dict]: entradas ``{'path': str, 'label': int}``
    """
    processed = []
    counters = {"Normal": 0, "Shoplifting": 0}

    videos = list(find_videos_recursively(source_root))
    for vid_path in tqdm(videos, desc=f"Processando simples: {prefix}"):
        vid_path_lower = vid_path.lower().replace("\\", "/")
        label = None
        for key, val in mapping_folder_to_label.items():
            if f"/{key.lower()}/" in vid_path_lower:
                label = val
                break
        if label is None:
            print(f"Aviso: não foi possível determinar label para {vid_path}. Pulando.")
            continue

        out_dir_name = "Shoplifting" if label == 1 else "Normal"
        out_dir = os.path.join(dest_root, out_dir_name)
        safe_makedirs(out_dir)

        count = counters[out_dir_name]
        out_name = f"{prefix}_{count:04d}.mp4"
        out_path = os.path.join(out_dir, out_name)
        counters[out_dir_name] += 1

        tmp_list = os.path.join(out_dir, TMP_LIST_FILENAME)
        write_ffmpeg_file_list([vid_path], tmp_list)

        ok, err = run_ffmpeg_concat_and_standardize(tmp_list, out_path, fps=fps, w=w, h=h)
        if not ok:
            print(f"Erro ao processar {vid_path} -> {out_path}: {err}")
        else:
            processed.append({"path": os.path.abspath(out_path), "label": label})

        try:
            if os.path.exists(tmp_list):
                os.remove(tmp_list)
        except Exception:
            pass

    return processed


# ── Geração de manifest ──────────────────────────────────────────────────────

def generate_manifest(output_root, manifest_filename=MANIFEST_FILENAME):
    """
    Gera manifest.csv varrendo as pastas Normal/ e Shoplifting/ dentro de output_root.
    """
    manifest_path = os.path.join(output_root, manifest_filename)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for label_name, label_id in [("Normal", 0), ("Shoplifting", 1)]:
            class_dir = os.path.join(output_root, label_name)
            if not os.path.isdir(class_dir):
                continue
            for root, _, files in os.walk(class_dir):
                for fn in sorted(files):
                    if fn.lower().endswith(".mp4"):
                        writer.writerow([os.path.abspath(os.path.join(root, fn)), label_id])
    print(f"Manifest gerado em: {manifest_path}")
    return manifest_path
