#!/usr/bin/env python3
"""
Função principal parametrizada para o TimeSformer preprocessing.
"""

def main_preprocess_timesformer(config_data):
    """
    Função principal parametrizada para preprocessamento TimeSformer.
    
    Args:
        config_data: Dicionário com configurações
    """
    from process_and_standarize_data import (
        ensure_ffmpeg_exists, safe_makedirs, load_annotations as load_dcsass_annotations,
        identify_event_blocks_with_context, process_simple_dataset,
        write_ffmpeg_file_list, run_ffmpeg_concat_and_standardize
    )
    import os
    import csv
    from tqdm import tqdm
    
    ensure_ffmpeg_exists()
    
    # Cria diretórios de saída
    safe_makedirs(config_data['output']['root'])
    safe_makedirs(os.path.join(config_data['output']['root'], 'Normal'))
    safe_makedirs(os.path.join(config_data['output']['root'], 'Shoplifting'))
    
    manifest_entries = []
    
    # 1) Processar DCSASS (lógica complexa de blocos + contexto)
    print("=== Processando DCSASS (blocos + contexto) ===")
    annotations = load_dcsass_annotations(config_data['datasets']['dcsass_annotations'])
    
    if annotations is not None:
        blocks = identify_event_blocks_with_context(config_data['datasets']['dcsass_root'], annotations)
        print(f"Total de blocos identificados: {len(blocks)}")
        
        # Processa blocos e cria mp4s padronizados
        counter_norm = 0
        counter_shop = 0
        
        for idx, blk in enumerate(tqdm(blocks, desc='DCSASS: processando blocos')):
            label = blk['label']
            clip_paths = blk['clip_paths']
            
            out_dir = os.path.join(config_data['output']['root'], 'Shoplifting' if label == 1 else 'Normal')
            safe_makedirs(out_dir)
            
            if label == 1:
                out_basename = f"{config_data['prefixes']['dcsass']}_shoplifting_{counter_shop:04d}.mp4"
                counter_shop += 1
            else:
                out_basename = f"{config_data['prefixes']['dcsass']}_normal_{counter_norm:04d}.mp4"
                counter_norm += 1
            
            out_path = os.path.join(out_dir, out_basename)
            
            # Escrever lista temporária
            tmp_list = os.path.join(out_dir, config_data['tmp_list_filename'])
            write_ffmpeg_file_list(clip_paths, tmp_list)
            
            ok, err = run_ffmpeg_concat_and_standardize(
                tmp_list, out_path,
                fps=config_data['output']['target_fps'],
                w=config_data['output']['target_w'],
                h=config_data['output']['target_h']
            )
            
            if not ok:
                print(f"Erro FFmpeg no bloco {idx} -> {out_path}: {err}")
            else:
                manifest_entries.append({'path': os.path.abspath(out_path), 'label': label})
            
            # Cleanup
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
    
    # Função process_simple_dataset já atualizada para receber config_data
    try:
        processed_mnnit = process_simple_dataset(
            config_data['datasets']['mnnit_root'], 
            mnnit_map, 
            config_data['output']['root'], 
            config_data['prefixes']['mnnit'],
            config_data
        )
        manifest_entries.extend(processed_mnnit)
    except Exception as e:
        print(f"Erro ao processar MNNIT: {e}")
    
    # 3) Processar Dataset 2.0 (simples)
    print("\n=== Processando Dataset 2.0 (simples) ===")
    s2_map = {
        'normal': 0,
        'shoplifting': 1,
        'see and let': 0,
    }
    
    try:
        processed_s2 = process_simple_dataset(
            config_data['datasets']['s2_root'], 
            s2_map, 
            config_data['output']['root'], 
            config_data['prefixes']['s2'],
            config_data
        )
        manifest_entries.extend(processed_s2)
    except Exception as e:
        print(f"Erro ao processar Dataset 2.0: {e}")
    
    # 4) Criar manifest.csv final
    print("\n=== Gerando manifest final ===")
    manifest_path = os.path.join(config_data['output']['root'], config_data['manifest_filename'])
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label'])  # cabeçalho opcional
        # Varre Normal
        for root, _, files in os.walk(os.path.join(config_data['output']['root'], 'Normal')):
            for fn in files:
                if fn.lower().endswith('.mp4'):
                    writer.writerow([os.path.abspath(os.path.join(root, fn)), 0])
        # Varre Shoplifting
        for root, _, files in os.walk(os.path.join(config_data['output']['root'], 'Shoplifting')):
            for fn in files:
                if fn.lower().endswith('.mp4'):
                    writer.writerow([os.path.abspath(os.path.join(root, fn)), 1])
    
    print(f"Manifest gerado em: {manifest_path}")
    print("=== PROCESSAMENTO CONCLUÍDO ===")
    return True