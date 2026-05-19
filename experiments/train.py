import torch
import torch.nn as nn
import time


def train_epoch_standard(model, optimizer, train_loader, device, loss_fn=None):
    """Standard training epoch for SGD/Adam/AdamW/Lion/Muon."""
    model.train()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    step_losses = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        step_losses.append(batch_loss)
        total_loss += batch_loss * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += inputs.size(0)

    return {
        "train_loss": total_loss / total,
        "train_acc": correct / total,
        "step_losses": step_losses,
    }


def train_epoch_spectral(spectral_filter, train_loader, device, diagnostic_interval=50):
    """Training epoch for SpectralConsensus optimizer."""
    spectral_filter.model.train()
    total_loss, correct, total = 0.0, 0, 0
    step_losses = []
    diagnostics_log = []
    step = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss_val, diag = spectral_filter.step(inputs, targets)

        step_losses.append(loss_val)
        total_loss += loss_val * inputs.size(0)

        with torch.no_grad():
            outputs = spectral_filter.model(inputs)
            correct += (outputs.argmax(1) == targets).sum().item()
        total += inputs.size(0)

        if step % diagnostic_interval == 0:
            diagnostics_log.append({
                "step": step,
                "eigenvalues": diag["eigenvalues"],
                "k": diag["k"],
                "consensus_ratio": diag["consensus_ratio"],
            })
        step += 1

    return {
        "train_loss": total_loss / total,
        "train_acc": correct / total,
        "step_losses": step_losses,
        "diagnostics": diagnostics_log,
    }


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += inputs.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


def compute_epoch_diagnostics(model, train_loader, device, loss_fn=None, max_batches=5):
    """Compute eigenvalue diagnostics for non-spectral optimizers on a few batches."""
    from spectral_optimizer import compute_eigenvalue_diagnostics
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()
    all_eigenvalues = []

    for i, (inputs, targets) in enumerate(train_loader):
        if i >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            pass
        eigs = compute_eigenvalue_diagnostics(model, loss_fn, inputs, targets)
        all_eigenvalues.append(eigs)

    avg_eigs = [sum(e[j] for e in all_eigenvalues) / len(all_eigenvalues)
                for j in range(len(all_eigenvalues[0]))]
    return avg_eigs
