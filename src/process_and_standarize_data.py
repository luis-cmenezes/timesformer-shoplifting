#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unify_preprocess.py

Script unificado para pré-processamento dos datasets:
- DCSASS_Dataset (lógica de blocos + contexto)
- Shoplifting - MNNIT (simples)
- Shoplifting Dataset 2.0 (simples)

Saída:
dataset_processado/
    Normal/
    Shoplifting/
manifest.csv  (caminho_absoluto,label)

Requisitos:
- ffmpeg disponível no PATH
- pandas, tqdm

Uso:
Ajuste as constantes de caminho abaixo e execute:
python unify_preprocess.py
"""

import os
import csv
import subprocess
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def get_default_config():
    """
    Retorna configuração padrão caso não seja fornecida via arquivo.
    """
    return {
        'datasets': {
            'dcsass_root': '/home/luis/tcc/datasets/DCSASS_Dataset',
            'dcsass_annotations': '/home/luis/tcc/datasets/DCSASS_Dataset/Shoplifting.csv',
            'mnnit_root': '/home/luis/tcc/datasets/Shoplifting - MNNIT',
            's2_root': '/home/luis/tcc/datasets/Shoplifting Dataset 2.0',
        },
        'output': {
            'root': '/home/luis/tcc/timesformer/data/standarized',
            'target_fps': 25,
            'target_w': 224,
            'target_h': 224,
        },
        'prefixes': {
            'dcsass': 'dcsass',
            'mnnit': 'mnnit',
            's2': 's2',
        },
        'tmp_list_filename': 'tmp_ffmpeg_file_list.txt',
        'manifest_filename': 'manifest.csv'
    }

def ensure_ffmpeg_exists():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale o ffmpeg e tente novamente.") from e

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)

def run_ffmpeg_concat_and_standardize(input_file_list, output_path, fps=25, w=224, h=224):
    """
    Concatena a lista de arquivos (concat demuxer), padroniza FPS e resolução com scale+pad,
    e escreve em output_path (mp4).
    
    input_file_list: caminho para um arquivo de texto com linhas "file '/abs/path/to/file'"
    output_path: caminho completo do vídeo final
    fps: taxa de frames de saída
    w, h: largura e altura alvo (pixels)
    """
    ratio = float(w) / float(h)
    
    # Escapando corretamente para o FFmpeg interpretar a expressão do scale e pad
    # Note as aspas simples internas
    scale_part = f"scale='if(gt(iw/ih,{ratio}),{w},-2)':'if(gt(iw/ih,{ratio}),-2,{h})'"
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
        str(output_path)
    ]

    try:
        # shell=False garante que os parênteses e vírgulas dentro do filtro sejam interpretados corretamente
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
    with open(file_path, 'w', encoding='utf-8') as f:
        for p in clip_paths:
            # FFmpeg espera: file '/absolute/path'
            f.write(f"file '{os.path.abspath(p)}'\n")

# -------------------------
# DCSASS: identificação de blocos com contexto (mesma lógica do seu exemplo)
# -------------------------
def load_annotations(annotations_path):
    try:
        df = pd.read_csv(annotations_path, header=None, names=['clip_name_base', 'category', 'label'])
        df.set_index('clip_name_base', inplace=True)
        print(f"Anotações carregadas de: {annotations_path}")
        return df
    except FileNotFoundError:
        print(f"Arquivo de anotações não encontrado em {annotations_path}")
        return None

def identify_event_blocks_with_context(dataset_root, annotations_df):
    """
    Itera sobre o dataset, identifica blocos de clipes contíguos com o mesmo rótulo
    e adiciona clipes de contexto (anterior/posterior) aos blocos de Shoplifting.
    """
    print("Identificando blocos de eventos e adicionando contexto...")
    all_blocks = []
    situation_folders = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])

    for situation_name in tqdm(situation_folders, desc="Analisando Situações"):
        situation_path = os.path.join(dataset_root, situation_name)
        
        # Coleta e ordena todos os clipes e seus rótulos para a situação atual
        situation_clip_data = []
        try:
            sorted_clip_names = sorted(
                [f for f in os.listdir(situation_path) if f.endswith('.mp4')],
                key=lambda name: int(os.path.splitext(name)[0].split('_')[-1])
            )
            for clip_filename in sorted_clip_names:
                clip_name_base = os.path.splitext(clip_filename)[0]
                label = annotations_df.loc[clip_name_base, 'label']
                situation_clip_data.append({
                    'path': os.path.join(situation_path, clip_filename),
                    'label': label
                })
        except (ValueError, IndexError, KeyError) as e:
            print(f"\nAviso: Problema ao processar clipes em '{situation_name}': {e}. Pulando situação.")
            continue
        
        if not situation_clip_data:
            continue

        # Itera sobre a lista de clipes da situação para identificar os blocos
        i = 0
        while i < len(situation_clip_data):
            start_index = i
            current_label = situation_clip_data[start_index]['label']
            
            # Encontra o final do bloco contíguo
            end_index = start_index
            while end_index + 1 < len(situation_clip_data) and situation_clip_data[end_index + 1]['label'] == current_label:
                end_index += 1
            
            # Extrai os caminhos dos clipes para o bloco atual
            block_clip_paths = [d['path'] for d in situation_clip_data[start_index : end_index + 1]]

            # Se o bloco for de Shoplifting, adiciona os clipes vizinhos (se existirem)
            if current_label == 1:
                # Adiciona o clipe anterior, se não for o primeiro da situação
                if start_index > 0:
                    context_before_path = situation_clip_data[start_index - 1]['path']
                    block_clip_paths.insert(0, context_before_path)
                
                # Adiciona o clipe posterior, se não for o último da situação
                if end_index < len(situation_clip_data) - 1:
                    context_after_path = situation_clip_data[end_index + 1]['path']
                    block_clip_paths.append(context_after_path)

            all_blocks.append({'label': current_label, 'clip_paths': block_clip_paths})
            
            # Pula para o início do próximo bloco
            i = end_index + 1
            
    print(f"Identificação concluída. Total de {len(all_blocks)} blocos de eventos encontrados.")
    return all_blocks

# -------------------------
# Processamento simplificado para os outros datasets
# -------------------------
def find_videos_recursively(root_dir):
    exts = ('.mp4', '.avi', '.mov', '.mkv')
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)

def process_simple_dataset(source_root, mapping_folder_to_label, dest_root, prefix):
    """
    Processa vídeos simples (MNNIT / Dataset 2.0), padroniza FPS e resolução,
    salva em pasta de saída baseada no mapping_folder_to_label e nomeia como <prefix>_<count>.mp4

    source_root: pasta raiz do dataset
    mapping_folder_to_label: dict de pasta (basename) -> label (0/1)
        Ex.: {'Normal': 0, 'shoplifting': 1, 'see and let': 0}
    dest_root: pasta raiz de saída
    prefix: string para prefixar os arquivos
    """
    processed = []

    # Inicializar contadores por label/pasta de saída
    counters = {'Normal': 0, 'Shoplifting': 0}

    for vid_path in tqdm(list(find_videos_recursively(source_root)), desc=f'Processando simples: {prefix}'):
        # determinar label a partir do caminho
        vid_path_lower = vid_path.lower()
        label = None
        for key, val in mapping_folder_to_label.items():
            # Verifica se a pasta exata está no caminho (case-insensitive)
            if f"/{key.lower()}/" in vid_path_lower.replace("\\", "/"):
                label = val
                break
        if label is None:
            print(f"Aviso: não foi possível determinar label para {vid_path}. Pulando.")
            continue

        out_dir_name = 'Shoplifting' if label == 1 else 'Normal'
        out_dir = os.path.join(dest_root, out_dir_name)
        safe_makedirs(out_dir)

        # Definir contador baseado na pasta de saída
        count = counters[out_dir_name]
        out_name = f"{prefix}_{count:04d}.mp4"
        out_path = os.path.join(out_dir, out_name)

        # Atualizar contador
        counters[out_dir_name] += 1

        # Preparar lista temporária com um único arquivo (concat aceita um único também)
        tmp_list = os.path.join(out_dir, TMP_LIST_FILENAME)
        write_ffmpeg_file_list([vid_path], tmp_list)

        ok, err = run_ffmpeg_concat_and_standardize(tmp_list, out_path)
        if not ok:
            print(f"Erro ao processar {vid_path} -> {out_path}: {err}")
        else:
            processed.append({'path': os.path.abspath(out_path), 'label': label})

        # Remover arquivo temporário
        try:
            if os.path.exists(tmp_list):
                os.remove(tmp_list)
        except Exception:
            pass

    return processed

# -------------------------
# Função principal que orquestra tudo
# -------------------------
def main():
    ensure_ffmpeg_exists()
    safe_makedirs(OUTPUT_ROOT)
    safe_makedirs(os.path.join(OUTPUT_ROOT, 'Normal'))
    safe_makedirs(os.path.join(OUTPUT_ROOT, 'Shoplifting'))

    manifest_entries = []

    # 1) Processar DCSASS (lógica complexa)
    print("\n=== Processando DCSASS (blocos + contexto) ===")
    annotations = load_annotations(DCSASS_ANNOTATIONS_CSV)
    if annotations is not None:
        blocks = identify_event_blocks_with_context(DCSASS_ROOT, annotations)
        print(f"Total de blocos identificados: {len(blocks)}")

        # iterar blocos e criar mp4s padronizados
        counter_norm = 0
        counter_shop = 0
        
        for idx, blk in enumerate(tqdm(blocks, desc='DCSASS: processando blocos')):
            label = blk['label']
            clip_paths = blk['clip_paths']
            out_dir = os.path.join(OUTPUT_ROOT, 'Shoplifting' if label == 1 else 'Normal')
            safe_makedirs(out_dir)

            if label == 1:
                out_basename = f"{PREFIX_DCSASS}_shoplifting_{counter_shop:04d}.mp4"
                counter_shop += 1
            else:
                out_basename = f"{PREFIX_DCSASS}_normal_{counter_norm:04d}.mp4"
                counter_norm += 1

            out_path = os.path.join(out_dir, out_basename)

            # escrever lista temporária (no diretório de saída para facilitar permissões)
            tmp_list = os.path.join(out_dir, TMP_LIST_FILENAME)
            write_ffmpeg_file_list(clip_paths, tmp_list)

            ok, err = run_ffmpeg_concat_and_standardize(tmp_list, out_path)
            if not ok:
                print(f"Erro FFmpeg no bloco {idx} -> {out_path}: {err}")
            else:
                manifest_entries.append({'path': os.path.abspath(out_path), 'label': label})

            # cleanup
            try:
                if os.path.exists(tmp_list):
                    os.remove(tmp_list)
            except Exception:
                pass
    else:
        print("Pular DCSASS (anotações não carregadas).")

    # 2) Processar MNNIT (lógica simples)
    print("\n=== Processando MNNIT (simples) ===")
    mnnit_map = {
        'Normal': 0,
        'Shoplifting': 1,
    }
    processed_mnnit = process_simple_dataset(MNNIT_ROOT, mnnit_map, OUTPUT_ROOT, PREFIX_MNNIT)
    manifest_entries.extend(processed_mnnit)

    # 3) Processar Dataset 2.0 (simples)
    print("\n=== Processando Dataset 2.0 (simples) ===")
    s2_map = {
        'normal': 0,
        'shoplifting': 1,
        'see and let': 0,
    }
    processed_s2 = process_simple_dataset(S2_ROOT, s2_map, OUTPUT_ROOT, PREFIX_S2)
    manifest_entries.extend(processed_s2)

    # 4) Criar manifest.csv final varrendo as pastas (garante inclusão de tudo)
    print("\n=== Gerando manifest final ===")
    manifest_path = os.path.join(OUTPUT_ROOT, MANIFEST_FILENAME)
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])  # cabeçalho opcional
        # Varre Normal
        for root, _, files in os.walk(os.path.join(OUTPUT_ROOT, 'Normal')):
            for fn in files:
                if fn.lower().endswith('.mp4'):
                    writer.writerow([os.path.abspath(os.path.join(root, fn)), 0])
        # Varre Shoplifting
        for root, _, files in os.walk(os.path.join(OUTPUT_ROOT, 'Shoplifting')):
            for fn in files:
                if fn.lower().endswith('.mp4'):
                    writer.writerow([os.path.abspath(os.path.join(root, fn)), 1])

    print(f"Manifest gerado em: {manifest_path}")
    print("=== PROCESSAMENTO CONCLUÍDO ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessa datasets para TimeSformer")
    parser.add_argument(
        "--config",
        type=str,
        help="Arquivo de configuração YAML"
    )
    parser.add_argument(
        "--dcsass-root",
        type=str,
        help="Caminho para DCSASS dataset"
    )
    parser.add_argument(
        "--dcsass-annotations",
        type=str,
        help="Caminho para arquivo de anotações DCSASS"
    )
    parser.add_argument(
        "--mnnit-root",
        type=str,
        help="Caminho para MNNIT dataset"
    )
    parser.add_argument(
        "--s2-root",
        type=str,
        help="Caminho para Shoplifting Dataset 2.0"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        help="Diretório de saída para dados padronizados"
    )
    
    args = parser.parse_args()
    
    # Carrega configuração
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        config_data = {
            'datasets': {
                'dcsass_root': args.dcsass_root or yaml_config.get('data', {}).get('datasets', {}).get('dcsass', {}).get('root', ''),
                'dcsass_annotations': args.dcsass_annotations or yaml_config.get('data', {}).get('datasets', {}).get('dcsass', {}).get('annotations', ''),
                'mnnit_root': args.mnnit_root or yaml_config.get('data', {}).get('datasets', {}).get('mnnit', {}).get('root', ''),
                's2_root': args.s2_root or yaml_config.get('data', {}).get('datasets', {}).get('shoplifting_2', {}).get('root', ''),
            },
            'output': {
                'root': args.output_root or yaml_config.get('models', {}).get('timesformer', {}).get('data_dir', ''),
                'target_fps': yaml_config.get('preprocessing', {}).get('timesformer', {}).get('target_fps', 25),
                'target_w': yaml_config.get('preprocessing', {}).get('timesformer', {}).get('target_width', 224),
                'target_h': yaml_config.get('preprocessing', {}).get('timesformer', {}).get('target_height', 224),
            },
            'prefixes': yaml_config.get('preprocessing', {}).get('timesformer', {}).get('prefixes', {
                'dcsass': 'dcsass',
                'mnnit': 'mnnit', 
                's2': 's2'
            }),
            'tmp_list_filename': 'tmp_ffmpeg_file_list.txt',
            'manifest_filename': 'manifest.csv'
        }
    else:
        # Usar configuração padrão se não fornecida
        config_data = get_default_config()
        
        if args.dcsass_root:
            config_data['datasets']['dcsass_root'] = args.dcsass_root
        if args.dcsass_annotations:
            config_data['datasets']['dcsass_annotations'] = args.dcsass_annotations
        if args.mnnit_root:
            config_data['datasets']['mnnit_root'] = args.mnnit_root
        if args.s2_root:
            config_data['datasets']['s2_root'] = args.s2_root
        if args.output_root:
            config_data['output']['root'] = args.output_root
        
        print("  Usando configuração padrão. Forneça --config para customizar.")
    
    # Cria diretórios de saída
    Path(config_data['output']['root']).mkdir(parents=True, exist_ok=True)
    
    success = main_preprocess_timesformer(config_data)
    exit(0 if success else 1)
