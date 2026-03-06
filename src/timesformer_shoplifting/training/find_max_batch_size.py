"""
Determinação automática do batch_size máximo para o modelo TimeSformer.

Realiza uma busca binária tentando forward + backward passes com tensores
sintéticos de tamanho crescente até detectar um erro de CUDA OOM.
O batch_size seguro retornado é ``int(0.8 * max_batch_size)``.
"""

import gc
import torch
import torch.nn as nn

from timesformer_shoplifting.models.model_utils import (
    get_model_and_processor,
    set_freeze_strategy,
)

# Dimensões padrão usadas pelo pipeline TimeSformer
_HEIGHT = 224
_WIDTH = 224
_CHANNELS = 3


def _try_batch_size(
    batch_size: int,
    model: nn.Module,
    num_frames: int,
    device: torch.device,
    use_fp16: bool = True,
) -> bool:
    """Tenta executar um forward + backward com *batch_size* amostras sintéticas.

    Retorna ``True`` se bem-sucedido, ``False`` se ocorrer OOM.
    """
    try:
        # TimeSformer espera pixel_values com shape (B, T, C, H, W)
        pixel_values = torch.randn(
            batch_size, num_frames, _CHANNELS, _HEIGHT, _WIDTH,
            device=device,
            dtype=torch.float16 if use_fp16 else torch.float32,
        )
        labels = torch.randint(0, 2, (batch_size,), device=device)

        if use_fp16:
            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            # Backward em fp32 para escala de gradientes
            loss.backward()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return False
        raise
    finally:
        del pixel_values, labels
        gc.collect()
        torch.cuda.empty_cache()


def find_max_batch_size(
    model_name: str = "facebook/timesformer-base-finetuned-k400",
    freeze_strategy: str = "unfreeze_head",
    num_frames: int = 8,
    start_batch_size: int = 128,
    use_fp16: bool = True,
    device: torch.device | None = None,
) -> int:
    """Encontra o batch_size máximo que cabe na GPU para o TimeSformer.

    Usa busca binária entre 1 e *start_batch_size*.

    Args:
        model_name: Identificador HuggingFace do checkpoint base.
        freeze_strategy: ``"unfreeze_head"`` ou ``"unfreeze_all"``.
        num_frames: Número de frames amostrados por vídeo.
        start_batch_size: Limite superior inicial para a busca binária.
        use_fp16: Se ``True``, usa mixed-precision (como o Trainer faz).
        device: Dispositivo CUDA. Se ``None``, usa ``cuda:0``.

    Returns:
        Batch_size seguro (``int(0.8 * max_batch_size)``), mínimo 1.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("[find_max_batch_size] Sem GPU detectada — retornando batch_size=1.")
        return 1

    print(f"[find_max_batch_size] Procurando batch_size máximo para TimeSformer "
          f"(model={model_name}, strategy={freeze_strategy}, "
          f"frames={num_frames}, fp16={use_fp16}) ...")

    # --- Instancia modelo ---
    model, _processor = get_model_and_processor(
        model_name, num_labels=2, num_frames=num_frames,
    )
    set_freeze_strategy(model, strategy=freeze_strategy)
    model.to(device)
    model.train()

    # --- Busca binária ---
    low, high = 1, start_batch_size
    max_ok = 0

    while low <= high:
        mid = (low + high) // 2
        print(f"  Tentando batch_size={mid} ...", end=" ", flush=True)

        model.zero_grad()
        success = _try_batch_size(mid, model, num_frames, device, use_fp16)

        if success:
            print("OK")
            max_ok = mid
            low = mid + 1
        else:
            print("OOM")
            high = mid - 1

    # --- Cleanup ---
    del model
    gc.collect()
    torch.cuda.empty_cache()

    safe_batch_size = max(1, int(0.8 * max_ok))
    print(f"[find_max_batch_size] max_batch_size={max_ok} → "
          f"safe batch_size (×0.8) = {safe_batch_size}")
    return safe_batch_size
