import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import decord
from torchvision.transforms import v2 as transforms

# Configura o Decord para utilizar tensores PyTorch diretamente, evitando cópias de memória desnecessárias.
# Em ambientes com múltiplas GPUs, a leitura via CPU é geralmente mais estável para o DataLoader.
decord.bridge.set_bridge('torch')


class VideoAugmentation:
    """
    Implementa data augmentation para vídeos RGB, idêntico ao I3D.
    Aplicado apenas durante treinamento.
    """
    
    def __init__(self, p_flip=0.5, color_jitter_params=None):
        """
        Args:
            p_flip (float): Probabilidade de horizontal flip (padrão: 0.5 = 50%)
            color_jitter_params (dict): Parâmetros para Color Jitter
                - brightness: float (padrão: 0.2 = ±20%)
                - contrast: float (padrão: 0.2 = ±20%)
                - saturation: float (padrão: 0.2 = ±20%)
                - hue: float (padrão: 0.1 = ±10%)
        """
        self.p_flip = p_flip
        
        if color_jitter_params is None:
            color_jitter_params = {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            }
        
        self.color_jitter = transforms.ColorJitter(**color_jitter_params)
    
    def __call__(self, video_tensor):
        """
        Aplica augmentation ao tensor de vídeo RGB.
        Args:
            video_tensor (torch.Tensor): Tensor (T, C, H, W)
        """
        # Horizontal Flip (50% de probabilidade)
        if random.random() < self.p_flip:
            video_tensor = torch.flip(video_tensor, dims=[-1])  # Flip no eixo W (largura)
        
        # Color Jitter aplicado ao tensor (T, C, H, W) inteiro de uma vez!
        # O v2 garante que o MESMO jitter seja aplicado a todos os frames,
        # mantendo a consistência temporal e evitando o efeito "pisca-pisca".
        video_tensor = self.color_jitter(video_tensor)
        
        return video_tensor


class SecurityVideoDataset(Dataset):
    """
    Dataset especializado para ingestão de vídeos de segurança.
    Mapeia a estrutura de pastas data/standarized/Normal e Shoplifting.
    """
    
    def __init__(
        self,
        root_dir,
        image_processor,
        num_frames=8,
        split="train",
        label_map=None,
        augmentation_p_flip=0.5,
        augmentation_color_jitter=None,
    ):
        """
        Args:
            root_dir (str): Caminho para data/standarized
            image_processor: O processador AutoImageProcessor do Hugging Face.
            num_frames (int): Número de quadros a serem amostrados (T). Padrão TimeSformer é 8.
            split (str): 'train', 'val' ou 'test'. Define a estratégia de amostragem.
            label_map (dict): Mapeamento opcional de string para int.
            augmentation_p_flip (float): Probabilidade de flip horizontal.
            augmentation_color_jitter (dict | None): Parâmetros do ColorJitter.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.split = split
        
        # Inicializa a augmentation (usada apenas em treino)
        if self.split == "train":
            cj_params = augmentation_color_jitter if augmentation_color_jitter is not None else {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1,
            }
            self.augmentation = VideoAugmentation(
                p_flip=augmentation_p_flip,
                color_jitter_params=cj_params,
            )
        else:
            self.augmentation = None
        
        # Definição automática de labels baseada na estrutura de pastas
        if label_map:
            self.label_map = label_map
        else:
            self.label_map = {"Normal": 0, "Shoplifting": 1}
            
        self.video_paths = []
        self.labels = []
        self._build_index()

    def _build_index(self):
        """
        Constrói o índice de arquivos. Ignora arquivos que não sejam vídeos.
        Lê recursivamente as pastas definidas em label_map.
        """
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        # print(f"Indexando dataset em {self.root_dir}...")
        
        for class_name, label_id in self.label_map.items():
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_path):
                print(f"AVISO: Diretório da classe {class_name} não encontrado em {class_path}")
                continue
                
            count = 0
            for root, _, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        self.video_paths.append(os.path.join(root, file))
                        self.labels.append(label_id)
                        count += 1
            # print(f"  Classe '{class_name}': {count} vídeos encontrados.")

    def _temporal_sampling(self, total_frames):
        """
        Implementa a Amostragem Uniforme Estratificada.
        Retorna os índices dos quadros a serem lidos.
        """
        # Garante que temos quadros suficientes; se não, repete o último
        if total_frames <= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            return indices
            
        # Divide o vídeo em num_frames segmentos
        seg_size = float(total_frames - 1) / self.num_frames
        indices = []
        
        for i in range(self.num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            
            if self.split == "train":
                # Aleatoriedade temporal para robustez durante o treino
                idx = np.random.randint(start, end + 1) if end > start else start
            else:
                # Determinismo (centro do segmento) para validação/teste
                idx = (start + end) // 2
                
            indices.append(idx)
            
        return np.array(indices).clip(0, total_frames - 1)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Inicializa o container de vídeo
            # num_threads=1 evita contenção de threads quando usado com DataLoader workers > 0
            vr = decord.VideoReader(video_path, num_threads=1)
            total_frames = len(vr)
            
            # Obtém índices baseados na estratégia temporal
            frame_indices = self._temporal_sampling(total_frames)
            
            # Leitura em batch (muito mais rápido que loop for)
            # Retorna tensor (T, H, W, C)
            video_data = vr.get_batch(frame_indices)
            
            # Permuta para (T, C, H, W) e normaliza para 0-1 float se necessário,
            # mas o ImageProcessor do HF espera lista de arrays ou tensor (T, C, H, W).
            # O decord retorna uint8. O Processor converte para float e normaliza (ImageNet stats).
            video_data = video_data.permute(0, 3, 1, 2) 
            
            # Aplicar Data Augmentation apenas durante treinamento
            if self.augmentation is not None:
                video_data = self.augmentation(video_data)

            # Aplicar o processador de imagem (Resize, CenterCrop, Normalize)
            # input deve ser lista de tensores ou numpy arrays
            inputs = self.image_processor(list(video_data), return_tensors="pt")
            
            # O output é um dicionário {'pixel_values': tensor(1, T, C, H, W)}
            # Removemos a dimensão de batch extra (1) criada pelo processor
            pixel_values = inputs.pixel_values.squeeze(0)
            
            return {
                "pixel_values": pixel_values, 
                "labels": torch.tensor(label, dtype=torch.long)
            }
            
        except Exception as e:
            # Estratégia de falha suave: imprime erro e retorna vídeo "preto"
            # Isso evita que o treinamento de 10 horas pare por causa de 1 arquivo corrompido.
            print(f"ERRO ao ler {video_path}: {e}")
            dummy_data = torch.zeros((self.num_frames, 3, 224, 224)) # Assumindo 224x224 padrão
            return {
                "pixel_values": dummy_data, 
                "labels": torch.tensor(label, dtype=torch.long)
            }