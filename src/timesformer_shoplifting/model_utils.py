from transformers import TimesformerForVideoClassification, TimesformerConfig, AutoImageProcessor

def get_model_and_processor(model_name="facebook/timesformer-base-finetuned-k400", num_labels=2):
    """
    Carrega o modelo TimeSformer e prepara a cabeça de classificação para a tarefa binária.
    
    Args:
        model_name (str): Hugging Face Hub ID. 
            Exemplos: "facebook/timesformer-base-finetuned-k400"
                     "facebook/timesformer-base-finetuned-ssv2"
        num_labels (int): 2 para Normal vs Shoplifting.
    """
    
    print(f"Carregando configuração e processador para: {model_name}. Mensagem de loading do modelo:")
    print()
    print("="*80)
    
    # Processador responsável por: Resize(224), CenterCrop(224), Normalize(mean/std)
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    
    # Carregar configuração base
    config = TimesformerConfig.from_pretrained(model_name)
    
    # Ajuste de metadados de labels
    config.num_labels = num_labels
    config.id2label = {0: "Normal", 1: "Shoplifting"}
    config.label2id = {"Normal": 0, "Shoplifting": 1}
    
    # Carregamento do modelo
    # ignore_mismatched_sizes=True é FUNDAMENTAL.
    # O checkpoint original tem uma camada final (Kinetics-400).
    # Essa flag permite descartar a camada antiga e iniciar a nova aleatoriamente.
    model = TimesformerForVideoClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    print("="*80)
    print()
    return model, processor


def set_freeze_strategy(model, strategy):
    """
    Define a estratégia de congelamento de parâmetros do modelo.
    
    Args:
        model: Modelo TimeSformerForVideoClassification
        strategy (str): 
            - "unfreeze_head": Congela backbone, descongelado apenas o classificador (head)
            - "unfreeze_all": Descongela todo o modelo para Fine-Tuning completo
    """
    if strategy == "unfreeze_head":
        # Congela tudo primeiro
        for param in model.timesformer.parameters():
            param.requires_grad = False
        
        # Garante que a cabeça de classificação (classifier) esteja descongelada
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("- Backbone congelado. Apenas o classificador (head) será treinado.")
        
    elif strategy == "unfreeze_all":
        # Descongela tudo
        for param in model.parameters():
            param.requires_grad = True
        print("- Modelo totalmente descongelado para Fine-Tuning completo.")
    
    else:
        raise ValueError(f"strategy deve ser 'unfreeze_head' ou 'unfreeze_all'. Recebido: {strategy}")