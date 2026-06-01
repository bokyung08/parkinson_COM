from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset

from .data import Gait17Dataset, materialize_arrays, materialize_tabular
from .models import expected_score_from_logits, make_model, ordinal_focal_loss

LU_CLASSIFIERS = {"lu_ofddnet_official"}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "medae": float(median_absolute_error(y_true, y_pred)),
    }


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def sklearn_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "svr": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", SVR(C=1.0, epsilon=0.1))]),
        "rf": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestRegressor(n_estimators=300, random_state=random_state))]),
        "mlp_shallow": Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(64,), max_iter=1000, random_state=random_state)),
            ]
        ),
    }


def run_ml_fold(manifest_path: Path, train_idx: np.ndarray, val_idx: np.ndarray, args, model_name: str) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    x_train, y_train, _ = materialize_tabular(
        manifest_path,
        train_idx,
        args.max_len,
        args.ablation,
        scale_normalization=args.scale_normalization,
    )
    x_val, y_val, ids = materialize_tabular(
        manifest_path,
        val_idx,
        args.max_len,
        args.ablation,
        scale_normalization=args.scale_normalization,
    )
    model = clone(sklearn_models(args.random_state)[model_name])
    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - start
    infer_start = time.perf_counter()
    pred = model.predict(x_val).astype(np.float32)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0 / max(len(x_val), 1)
    row = {
        "model": model_name,
        "category": "Classical ML",
        "params": 0,
        "train_seconds": train_seconds,
        "inference_ms_per_sample": infer_ms,
        **regression_metrics(y_val, pred),
    }
    return row, y_val, pred, ids


def run_torch_fold(
    manifest_path: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args,
    model_name: str,
    checkpoint_path: Path | None = None,
    checkpoint_meta: dict | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    input_kind = "coords" if model_name in {"stgcn", *LU_CLASSIFIERS} else "hybrid"
    use_scale_aug = args.scale_aug_min != 1.0 or args.scale_aug_max != 1.0
    if use_scale_aug:
        x_val, y_val, ids = materialize_arrays(
            manifest_path,
            val_idx,
            args.max_len,
            args.ablation,
            input_kind,
            scale_normalization=args.scale_normalization,
        )
        train_ds = Gait17Dataset(
            manifest_path,
            train_idx,
            args.max_len,
            args.ablation,
            input_kind,
            scale_normalization=args.scale_normalization,
            scale_aug_min=args.scale_aug_min,
            scale_aug_max=args.scale_aug_max,
            random_state=args.random_state,
        )
        in_channels = int(x_val.shape[-1])
    else:
        x_train, y_train, _ = materialize_arrays(
            manifest_path,
            train_idx,
            args.max_len,
            args.ablation,
            input_kind,
            scale_normalization=args.scale_normalization,
        )
        x_val, y_val, ids = materialize_arrays(
            manifest_path,
            val_idx,
            args.max_len,
            args.ablation,
            input_kind,
            scale_normalization=args.scale_normalization,
        )
        train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        in_channels = int(x_train.shape[-1])
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = make_model(model_name, in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    val_tensor = torch.from_numpy(x_val).float().to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_val = float("inf")
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            xb, yb = batch[:2]
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            if model_name in LU_CLASSIFIERS:
                loss = ordinal_focal_loss(output, yb)
            else:
                loss = criterion(output, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(val_tensor)
            pred_val = expected_score_from_logits(out) if model_name in LU_CLASSIFIERS else out
            val_mae = torch.mean(torch.abs(pred_val - torch.from_numpy(y_val).float().to(device))).item()
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "input_kind": input_kind,
                "in_channels": int(in_channels),
                "ablation": args.ablation,
                "scale_normalization": args.scale_normalization,
                "scale_aug_min": float(args.scale_aug_min),
                "scale_aug_max": float(args.scale_aug_max),
                "max_len": int(args.max_len),
                "best_val_mae": float(best_val),
                "meta": checkpoint_meta or {},
            },
            checkpoint_path,
        )
    model.eval()
    infer_start = time.perf_counter()
    with torch.no_grad():
        out = model(val_tensor)
        pred = expected_score_from_logits(out) if model_name in LU_CLASSIFIERS else out
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0 / max(len(x_val), 1)
    category = {
        "temporal_cnn": "Deep Learning",
        "ours": "Proposed",
        "stgcn": "SOTA",
        "lu_ofddnet_official": "SOTA",
    }[model_name]
    row = {
        "model": model_name,
        "category": category,
        "params": parameter_count(model),
        "train_seconds": train_seconds,
        "inference_ms_per_sample": infer_ms,
        **regression_metrics(y_val, pred_np),
    }
    return row, y_val, pred_np, ids
