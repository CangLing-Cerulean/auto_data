"""Optional Transformers PatchTST, trained locally with data-only JSON weights."""
from __future__ import annotations

import time

import numpy as np

from .config import GateError


def _dependencies():
    try:
        import torch
        from transformers import PatchTSTConfig, PatchTSTForPrediction
    except ImportError as error:
        raise GateError("PATCHTST_DEPENDENCIES_REQUIRED") from error
    return torch, PatchTSTConfig, PatchTSTForPrediction


def train_patchtst(x, y, candidate, config):
    torch, model_config, model_type = _dependencies()
    settings = config.patchtst
    started = time.perf_counter()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        architecture = dict(num_input_channels=len(config.variables), context_length=candidate.input_steps,
                            prediction_length=config.horizon_steps, patch_length=settings.patch_length,
                            patch_stride=settings.patch_stride, d_model=settings.d_model,
                            num_hidden_layers=settings.num_hidden_layers,
                            num_attention_heads=settings.num_attention_heads, ffn_dim=2 * settings.d_model,
                            scaling=False, norm_type="layernorm", loss="mse",
                            attention_dropout=0.0, positional_dropout=0.0, path_dropout=0.0,
                            ff_dropout=0.0, head_dropout=0.0)
        model = model_type(model_config(**architecture)).cpu().train()
        inputs = torch.tensor(np.asarray(x), dtype=torch.float32)
        targets = torch.tensor(np.asarray(y), dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.learning_rate)
        generator = torch.Generator().manual_seed(config.seed)
        losses = []
        for _ in range(candidate.num_steps):
            indices = torch.randint(len(inputs), (min(candidate.batch_size, len(inputs)),), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss = model(past_values=inputs[indices], future_values=targets[indices]).loss
            if not torch.isfinite(loss):
                raise GateError("PATCHTST_NONFINITE_LOSS")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        return {"architecture": architecture,
                "state": {name: tensor.detach().cpu().tolist() for name, tensor in model.state_dict().items()},
                "training_evidence": {"backend": "torch-cpu", "pretrained": False,
                    "optimizer_steps": len(losses), "loss": "mse", "losses": losses,
                    "training_windows": len(inputs), "seed": config.seed,
                    "seconds": time.perf_counter() - started,
                    "parameters": sum(p.numel() for p in model.parameters())}}


def predict_patchtst(forecast, z):
    torch, model_config, model_type = _dependencies()
    if not hasattr(forecast, "_patchtst_model"):
        learned = forecast.payload["learned"]
        with torch.random.fork_rng(devices=[]):
            model = model_type(model_config(**learned["architecture"])).cpu()
        state = model.state_dict()
        if set(state) != set(learned["state"]):
            raise GateError("PATCHTST_WEIGHT_KEYS")
        restored = {name: torch.tensor(learned["state"][name], dtype=tensor.dtype)
                    for name, tensor in state.items()}
        if any(restored[name].shape != tensor.shape or not torch.isfinite(restored[name]).all()
               for name, tensor in state.items()):
            raise GateError("PATCHTST_WEIGHT_VALUES")
        model.load_state_dict(restored, strict=True)
        forecast._patchtst_model = model.eval()
    results = []
    with torch.inference_mode():
        for offset in range(0, len(z), forecast.candidate.batch_size):
            inputs = torch.tensor(z[offset:offset + forecast.candidate.batch_size], dtype=torch.float32)
            results.append(forecast._patchtst_model(past_values=inputs).prediction_outputs.cpu().numpy())
    return np.concatenate(results)
