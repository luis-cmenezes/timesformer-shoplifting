import torch
import torch.nn.functional as F
from transformers import TimesformerForVideoClassification, TimesformerConfig, AutoImageProcessor


def _interpolate_temporal_embeddings(
    model: TimesformerForVideoClassification,
    target_num_frames: int,
) -> None:
    """Interpola os ``time_embeddings`` pré-treinados para um novo número de frames.

    O checkpoint oficial do TimeSformer usa 8 frames.  Se o usuário quiser
    treinar com mais frames (ex.: 32, 64, 96), os 8 embeddings temporais
    originais são interpolados linearmente para ``target_num_frames`` posições,
    preservando o conhecimento temporal aprendido no pré-treino.

    A interpolação é feita **in-place** — o parâmetro ``time_embeddings``
    do modelo é substituído por um novo ``nn.Parameter`` com shape
    ``(1, target_num_frames, hidden_size)``.
    """
    old_time_emb = model.timesformer.embeddings.time_embeddings  # (1, old_T, D)
    old_T = old_time_emb.shape[1]

    if old_T == target_num_frames:
        return  # Nada a fazer

    print(f"  Interpolando time_embeddings de {old_T} → {target_num_frames} frames ...")

    # (1, old_T, D) → (1, D, old_T) para usar F.interpolate (espera dim espacial no fim)
    emb = old_time_emb.data.permute(0, 2, 1)  # (1, D, old_T)
    emb_interp = F.interpolate(emb, size=target_num_frames, mode="linear", align_corners=False)
    emb_interp = emb_interp.permute(0, 2, 1)  # (1, target_T, D)

    model.timesformer.embeddings.time_embeddings = torch.nn.Parameter(emb_interp)

    # Atualiza o config interno para refletir o novo num_frames
    model.config.num_frames = target_num_frames


def get_model_and_processor(
    model_name="facebook/timesformer-base-finetuned-k400",
    num_labels=2,
    num_frames=8,
):
    """
    Carrega o modelo TimeSformer e prepara a cabeça de classificação para a tarefa binária.

    Se ``num_frames`` for diferente do valor original do checkpoint (tipicamente 8),
    os ``time_embeddings`` são interpolados (e não reinicializados aleatoriamente),
    preservando o conhecimento temporal do pré-treino.

    Args:
        model_name (str): Hugging Face Hub ID.
            Exemplos: "facebook/timesformer-base-finetuned-k400"
                     "facebook/timesformer-base-finetuned-ssv2"
        num_labels (int): 2 para Normal vs Shoplifting.
        num_frames (int): Número de frames desejado para a entrada do modelo.
            Se > 8, os time_embeddings pré-treinados serão interpolados.
    """

    print(f"Carregando configuração e processador para: {model_name}. Mensagem de loading do modelo:")
    print()
    print("="*80)

    # Processador responsável por: Resize(224), CenterCrop(224), Normalize(mean/std)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    # --- Passo 1: Carregar o modelo com o num_frames ORIGINAL do checkpoint ---
    # Isso garante que os time_embeddings pré-treinados sejam carregados corretamente
    # (sem mismatch → sem reinicialização aleatória).
    config_original = TimesformerConfig.from_pretrained(model_name)
    original_num_frames = config_original.num_frames  # tipicamente 8

    config_original.num_labels = num_labels
    config_original.id2label = {0: "Normal", 1: "Shoplifting"}
    config_original.label2id = {"Normal": 0, "Shoplifting": 1}

    model = TimesformerForVideoClassification.from_pretrained(
        model_name,
        config=config_original,
        ignore_mismatched_sizes=True,  # necessário para a head de classificação
    )

    print("="*80)
    print()

    # --- Passo 2: Se o num_frames desejado difere, interpolar time_embeddings ---
    is_interpolated = False
    if num_frames != original_num_frames:
        is_interpolated = True
        _interpolate_temporal_embeddings(model, target_num_frames=num_frames)
        print(f"  Modelo configurado para {num_frames} frames (original: {original_num_frames})")
    else:
        print(f"  Modelo configurado para {num_frames} frames (padrão do checkpoint)")

    return model, processor, is_interpolated


def set_freeze_strategy(model, strategy, unfreeze_time_embeddings=False):
    """
    Define a estratégia de congelamento de parâmetros do modelo.
    
    Args:
        model: Modelo TimeSformerForVideoClassification
        strategy (str): 
            - "unfreeze_head": Congela backbone, descongelado apenas o classificador (head)
            - "unfreeze_all": Descongela todo o modelo para Fine-Tuning completo
        unfreeze_time_embeddings (bool): Se True e strategy == 'unfreeze_head',
            descongela os time_embeddings (útil quando foram interpolados para
            um num_frames diferente do original).
    """
    
    # Descongela time_embeddings se foram interpolados (num_frames diferente do original)
    if unfreeze_time_embeddings:
        model.timesformer.embeddings.time_embeddings.requires_grad = True
        print("- Numero de frames diferente do original. time_embeddings serão treinados.")

    if strategy == "unfreeze_head":
        # Congela tudo primeiro
        for param in model.timesformer.parameters():
            param.requires_grad = False
        
        # Garante que a cabeça de classificação (classifier) esteja descongelada
        for param in model.classifier.parameters():
            param.requires_grad = True

        print("- Backbone congelado. O classificador (head) será treinado.")
        
    elif strategy == "unfreeze_all":
        # Descongela tudo
        for param in model.parameters():
            param.requires_grad = True
        print("- Modelo totalmente descongelado para Fine-Tuning completo.")
    
    else:
        raise ValueError(f"strategy deve ser 'unfreeze_head' ou 'unfreeze_all'. Recebido: {strategy}")