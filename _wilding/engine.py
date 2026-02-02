import torch
import torch.nn as nn
from utils import get_padding_params, apply_padding, undo_padding, to_tensor

def run_inference(model, image_np, device):
    model.eval()
    with torch.no_grad():
        x = to_tensor(image_np, device)
        pads = get_padding_params(x.shape)
        
        x_padded = apply_padding(x, pads)
        logits = model(x_padded)
        
        logits = undo_padding(logits, pads)
        return torch.argmax(logits, dim=1).cpu().squeeze().numpy()

def run_training_step(model, optimizer, image_np, labels_np, device, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 1. Isolate Image: Detach from any graph history
    x = to_tensor(image_np, device).detach()
    
    # 2. Isolate Labels: CLONE is mandatory here to prevent 
    # C++ engine crashes when napari modifies the buffer mid-training
    y = torch.as_tensor(labels_np, device=device).long().clone().unsqueeze(0)
    
    pads = get_padding_params(x.shape)
    x_pad = apply_padding(x, pads).contiguous() # Ensure contiguous memory
    
    # Pad labels; ensure result is on correct device and contiguous
    y_pad = apply_padding(y.unsqueeze(1).float(), pads).squeeze(1).long().contiguous()

    last_loss = 0
    for _ in range(epochs):
        # 3. Use set_to_none=True for 2026 performance/stability
        optimizer.zero_grad(set_to_none=True)
        
        output = model(x_pad)
        loss = criterion(output, y_pad)
        
        # The crash occurs here if x_pad or y_pad were modified by napari
        loss.backward() 
        optimizer.step()
        last_loss = loss.item()
        
    return last_loss