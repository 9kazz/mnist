import torch
import torch.nn as nn

def generate_ideal_digit(
    model:  nn.Module,
    digit:  int,
    device: torch.device | str,
    iterations:    int   = 500,
    learning_rate: float = 0.001,
) -> torch.Tensor:
    model.eval()
    model.requires_grad_(False)

    # Start from random image
    image  = torch.rand(1, 1, 28, 28, device=device, requires_grad=True)
    target = torch.tensor([digit], dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([image], lr=learning_rate)

    for _ in range(iterations):
        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image.clamp_(0.0, 1.0)

    return image.detach().cpu()
