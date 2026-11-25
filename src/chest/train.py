"""Training helpers shared by the Stage 2 neural-model search."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def evaluate_nmse(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
) -> float:
    """Evaluate ratio-of-total-power NMSE across all examples and slots."""
    if inputs.shape != targets.shape:
        raise ValueError("inputs and targets must have matching shapes")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model.eval()
    error_power = torch.zeros((), dtype=torch.float64, device=inputs.device)
    target_power = torch.zeros((), dtype=torch.float64, device=targets.device)
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            batch_inputs = inputs[start : start + batch_size]
            batch_targets = targets[start : start + batch_size]
            prediction = model(batch_inputs)
            if prediction.shape != batch_targets.shape:
                raise ValueError("model output and targets must have matching shapes")
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("model produced non-finite predictions")
            error_power += torch.sum(
                (prediction - batch_targets).to(torch.float64) ** 2
            )
            target_power += torch.sum(batch_targets.to(torch.float64) ** 2)
    if target_power.item() <= 0:
        raise ValueError("targets must have positive total power")
    return float((error_power / target_power).item())


def fit_model(
    model: nn.Module,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[nn.Module, float, int]:
    """Train for a fixed budget and restore the lowest-validation-NMSE epoch."""
    if train_inputs.shape != train_targets.shape:
        raise ValueError("training inputs and targets must have matching shapes")
    if validation_inputs.shape != validation_targets.shape:
        raise ValueError("validation inputs and targets must have matching shapes")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if (
        train_inputs.device != train_targets.device
        or train_inputs.device != validation_inputs.device
        or train_inputs.device != validation_targets.device
    ):
        raise ValueError("training and validation tensors must share a device")

    model.to(train_inputs.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_nmse = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(
            train_inputs.shape[0],
            generator=generator,
        )
        for start in range(0, train_inputs.shape[0], batch_size):
            indices = permutation[start : start + batch_size].to(
                train_inputs.device
            )
            batch_inputs = train_inputs.index_select(0, indices)
            batch_targets = train_targets.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_inputs)
            if prediction.shape != batch_targets.shape:
                raise ValueError("model output and targets must have matching shapes")
            loss = loss_function(prediction, batch_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}"
                )
            loss.backward()
            optimizer.step()

        validation_nmse = evaluate_nmse(
            model,
            validation_inputs,
            validation_targets,
            batch_size,
        )
        if validation_nmse < best_nmse:
            best_nmse = validation_nmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_nmse, best_epoch
