from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from itertools import product
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from datasets import load_dataset
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16

SEED = 42
REPO_ID = "creative-graphic-design/GraphicDesignEvaluation"
CONFIGS = {
    "alignment": "absolute-human-alignment",
    "overlap": "absolute-human-overlap",
    "whitespace": "absolute-human-whitespace",
}
PERT_ORDER = ["none", "small", "medium", "large"]

OUT = Path("artifact/ViT_XGBoost_first_round")
TMP = Path("artifact/_tmp_graphic_design")
IMG_DIR = TMP / "images"

EXPERT_FEATURES = [
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "saturation_std",
    "contrast_rms",
    "colorfulness",
    "hue_entropy",
    "warm_color_ratio",
    "white_ratio",
    "black_ratio",
    "edge_density",
    "low_texture_ratio",
    "vertical_symmetry",
    "horizontal_symmetry",
    "saliency_center_distance",
    "saliency_thirds_distance",
    "left_right_balance",
    "top_bottom_balance",
    "border_saliency_ratio",
    "quadrant_saliency_entropy",
    "occupied_bbox_ratio",
    "text_region_ratio_proxy",
    "text_region_count_proxy",
    "text_mean_aspect_proxy",
]


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_float(v):
    if v is None:
        return np.nan
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def perturbation_name(ds, value):
    feat = ds.features["perturbation"]
    if hasattr(feat, "int2str") and isinstance(value, (int, np.integer)):
        return feat.int2str(int(value))
    return str(value).strip().lower()


def image_digest(img: Image.Image) -> str:
    arr = np.asarray(img.convert("RGB"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="PNG", optimize=True)


def build_dataset() -> pd.DataFrame:
    if TMP.exists():
        shutil.rmtree(TMP)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    image_hashes: dict[str, str] = {}

    for principle, config in CONFIGS.items():
        print(f"[data] loading {config}")
        ds = load_dataset(REPO_ID, config, split="train")
        if len(ds) != 400:
            raise RuntimeError(f"{config}: expected 400 rows, got {len(ds)}")

        for idx, row in enumerate(ds):
            image_id = str(row["image_id"]).strip()
            pert = perturbation_name(ds, row["perturbation"])
            avg = safe_float(row["avg"])
            scores = [safe_float(x) for x in row["scores"]]

            if pert not in PERT_ORDER:
                raise RuntimeError(f"Unexpected perturbation {pert}")

            if pert == "none":
                sample_id = f"{image_id}__original"
                image_category = "original"
                relpath = Path("original") / f"{image_id}.png"
            elif principle == "alignment":
                sample_id = f"{image_id}__alignment__{pert}"
                image_category = "alignment_perturbed"
                relpath = Path("alignment") / pert / f"{image_id}.png"
            else:
                sample_id = f"{image_id}__overlap_whitespace__{pert}"
                image_category = "overlap_whitespace_perturbed"
                relpath = Path("overlap_whitespace") / pert / f"{image_id}.png"

            rec = records.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "group_id": image_id,
                    "image_path": str((IMG_DIR / relpath).as_posix()),
                    "image_category": image_category,
                    "perturbation": pert,
                    "alignment_score": np.nan,
                    "overlap_score": np.nan,
                    "whitespace_score": np.nan,
                },
            )
            rec[f"{principle}_score"] = avg

            digest = image_digest(row["image"])
            if sample_id in image_hashes:
                if image_hashes[sample_id] != digest:
                    raise RuntimeError(
                        f"Image content mismatch for merged sample {sample_id} "
                        f"between principle configs"
                    )
            else:
                image_hashes[sample_id] = digest
                save_image(row["image"], IMG_DIR / relpath)

    df = pd.DataFrame(records.values())
    if len(df) != 700:
        raise RuntimeError(f"Expected 700 unique images, got {len(df)}")
    if df["group_id"].nunique() != 100:
        raise RuntimeError(f"Expected 100 base designs, got {df['group_id'].nunique()}")

    # Overall derived score for original designs only.
    mask_original = df["perturbation"].eq("none")
    df["overall_score"] = np.nan
    df.loc[mask_original, "overall_score"] = df.loc[
        mask_original, ["alignment_score", "overlap_score", "whitespace_score"]
    ].mean(axis=1)

    # Same group split as the delivered 700-image package.
    groups = sorted(df["group_id"].unique().tolist())
    rng = random.Random(SEED)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train+n_val])
    df["split"] = df["group_id"].map(
        lambda g: "train" if g in train_groups else ("val" if g in val_groups else "test")
    )

    df = df.sort_values(["split", "group_id", "image_category", "perturbation", "sample_id"]).reset_index(drop=True)
    print("[data] rows:", len(df), "groups:", df["group_id"].nunique())
    print("[data] split:", df["split"].value_counts().to_dict())
    print("[data] labels:", df[["alignment_score","overlap_score","whitespace_score"]].notna().sum().to_dict())
    return df


def entropy_normalized(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def saliency_map(gray: np.ndarray) -> np.ndarray:
    # Prefer OpenCV spectral-residual saliency; fall back to gradient energy.
    try:
        if hasattr(cv2, "saliency"):
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, m = sal.computeSaliency(gray)
            if ok:
                m = np.asarray(m, dtype=np.float32)
                m = np.clip(m, 0, 1)
                return m
    except Exception:
        pass

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    mx = float(m.max())
    return m / mx if mx > 1e-6 else np.zeros_like(m, dtype=np.float32)


def text_proxy(gray: np.ndarray):
    # Lightweight typography-density proxy based on morphological grouping.
    h, w = gray.shape
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=1
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = h * w
    boxes = []
    aspects = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area_ratio = (ww * hh) / max(total_area, 1)
        aspect = ww / max(hh, 1)
        if (
            0.00025 <= area_ratio <= 0.12
            and ww >= 6
            and hh >= 3
            and 1.15 <= aspect <= 30
        ):
            boxes.append((x, y, ww, hh))
            aspects.append(aspect)

    area_sum = sum(ww * hh for _, _, ww, hh in boxes)
    return (
        min(area_sum / max(total_area, 1), 1.0),
        float(len(boxes)),
        float(np.mean(aspects)) if aspects else 0.0,
    )


def professional_features(path: str) -> dict[str, float]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    # Normalize compute cost while preserving the full composition.
    h0, w0 = img.shape[:2]
    max_side = 512
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w0*scale)), int(round(h0*scale))), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = gray.shape

    v = hsv[:, :, 2].astype(np.float32) / 255.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    hue = hsv[:, :, 0].astype(np.float32)  # [0,179]

    # Colorfulness, Hasler-Susstrunk.
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = math.sqrt(float(rg.std())**2 + float(yb.std())**2) + 0.3 * math.sqrt(float(rg.mean())**2 + float(yb.mean())**2)
    colorfulness /= 255.0

    sat_mask = s > 0.08
    if sat_mask.any():
        hist, _ = np.histogram(hue[sat_mask], bins=36, range=(0, 180))
        hue_ent = entropy_normalized(hist)
        warm = (((hue <= 30) | (hue >= 165)) & sat_mask).sum() / sat_mask.sum()
    else:
        hue_ent = 0.0
        warm = 0.0

    white = np.all(rgb >= 235, axis=2)
    black = np.all(rgb <= 25, axis=2)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    low_texture_ratio = float((grad_mag < 18.0).mean())

    # Symmetry scores in [0,1].
    half_w = w // 2
    if half_w > 0:
        left = gray[:, :half_w].astype(np.float32)
        right = np.fliplr(gray[:, w-half_w:]).astype(np.float32)
        vertical_sym = 1.0 - float(np.mean(np.abs(left-right))) / 255.0
    else:
        vertical_sym = 0.0

    half_h = h // 2
    if half_h > 0:
        top = gray[:half_h, :].astype(np.float32)
        bottom = np.flipud(gray[h-half_h:, :]).astype(np.float32)
        horizontal_sym = 1.0 - float(np.mean(np.abs(top-bottom))) / 255.0
    else:
        horizontal_sym = 0.0

    sal = saliency_map(gray)
    sal = np.maximum(sal.astype(np.float64), 0)
    total = float(sal.sum()) + 1e-12
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((sal * xx).sum() / total) / max(w-1, 1)
    cy = float((sal * yy).sum() / total) / max(h-1, 1)
    center_dist = math.sqrt((cx-0.5)**2 + (cy-0.5)**2) / math.sqrt(0.5**2 + 0.5**2)

    thirds = [(1/3,1/3),(2/3,1/3),(1/3,2/3),(2/3,2/3)]
    thirds_dist = min(math.sqrt((cx-tx)**2 + (cy-ty)**2) for tx,ty in thirds) / math.sqrt(2)

    left_mass = float(sal[:, :w//2].sum())
    right_mass = float(sal[:, w//2:].sum())
    top_mass = float(sal[:h//2, :].sum())
    bottom_mass = float(sal[h//2:, :].sum())
    lr_balance = 1.0 - abs(left_mass-right_mass)/total
    tb_balance = 1.0 - abs(top_mass-bottom_mass)/total

    bh = max(1, int(round(h*0.10)))
    bw = max(1, int(round(w*0.10)))
    border_mask = np.zeros((h,w), dtype=bool)
    border_mask[:bh,:] = True
    border_mask[-bh:,:] = True
    border_mask[:,:bw] = True
    border_mask[:,-bw:] = True
    border_ratio = float(sal[border_mask].sum() / total)

    q = np.array([
        sal[:h//2,:w//2].sum(),
        sal[:h//2,w//2:].sum(),
        sal[h//2:,:w//2].sum(),
        sal[h//2:,w//2:].sum(),
    ], dtype=np.float64)
    quadrant_ent = entropy_normalized(q)

    # Occupied bounding box estimated from non-white pixels and edges.
    occupied = (~white) | (edges > 0)
    ys, xs = np.where(occupied)
    if len(xs):
        bbox_ratio = float((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1)/(h*w))
    else:
        bbox_ratio = 0.0

    text_ratio, text_count, text_aspect = text_proxy(gray)

    return {
        "brightness_mean": float(v.mean()),
        "brightness_std": float(v.std()),
        "saturation_mean": float(s.mean()),
        "saturation_std": float(s.std()),
        "contrast_rms": float(gray.astype(np.float32).std()/255.0),
        "colorfulness": float(colorfulness),
        "hue_entropy": float(hue_ent),
        "warm_color_ratio": float(warm),
        "white_ratio": float(white.mean()),
        "black_ratio": float(black.mean()),
        "edge_density": float(edge_density),
        "low_texture_ratio": float(low_texture_ratio),
        "vertical_symmetry": float(np.clip(vertical_sym, 0, 1)),
        "horizontal_symmetry": float(np.clip(horizontal_sym, 0, 1)),
        "saliency_center_distance": float(center_dist),
        "saliency_thirds_distance": float(thirds_dist),
        "left_right_balance": float(np.clip(lr_balance, 0, 1)),
        "top_bottom_balance": float(np.clip(tb_balance, 0, 1)),
        "border_saliency_ratio": float(border_ratio),
        "quadrant_saliency_entropy": float(quadrant_ent),
        "occupied_bbox_ratio": float(bbox_ratio),
        "text_region_ratio_proxy": float(text_ratio),
        "text_region_count_proxy": float(text_count),
        "text_mean_aspect_proxy": float(text_aspect),
    }


def extract_professional(df: pd.DataFrame) -> pd.DataFrame:
    print("[expert] extracting professional visual-communication features")
    rows = []
    for i, row in df.iterrows():
        f = professional_features(row["image_path"])
        f["sample_id"] = row["sample_id"]
        rows.append(f)
        if (i + 1) % 100 == 0:
            print(f"[expert] {i+1}/{len(df)}")
    feat = pd.DataFrame(rows)
    missing = feat[EXPERT_FEATURES].isna().sum().sum()
    if missing:
        raise RuntimeError(f"Professional features contain {missing} NaNs")
    return feat


class ImagePathDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        with Image.open(r["image_path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, r["sample_id"]


def extract_vit(df: pd.DataFrame, batch_size: int = 16):
    print("[vit] loading pretrained ViT-B/16 ImageNet-1K weights")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()
    model.eval()

    torch.set_num_threads(max(1, min(4, (__import__("os").cpu_count() or 2))))
    device = torch.device("cpu")
    model.to(device)

    ds = ImagePathDataset(df, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_f = []
    all_ids = []
    with torch.inference_mode():
        seen = 0
        for xb, ids in loader:
            out = model(xb.to(device))
            all_f.append(out.cpu().numpy().astype(np.float32))
            all_ids.extend(list(ids))
            seen += len(ids)
            print(f"[vit] {seen}/{len(df)}")

    features = np.vstack(all_f)
    if features.shape != (len(df), 768):
        raise RuntimeError(f"Unexpected ViT feature shape: {features.shape}")
    return all_ids, features


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    try:
        pearson = float(pearsonr(y_true, y_pred).statistic)
    except Exception:
        pearson = np.nan
    try:
        spear = float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        spear = np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": pearson, "Spearman": spear}


def tune_xgb(X_train, y_train, X_val, y_val):
    params_grid = []
    for max_depth, lr, n_est in product([2, 3], [0.03, 0.07], [200, 400]):
        params_grid.append(
            dict(
                max_depth=max_depth,
                learning_rate=lr,
                n_estimators=n_est,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.0,
            )
        )

    best = None
    best_rmse = float("inf")
    for p in params_grid:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=2,
            tree_method="hist",
            **p,
        )
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best = p
    return best, best_rmse


def fit_final_xgb(X_train, y_train, X_val, y_val, params):
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=2,
        tree_method="hist",
        **params,
    )
    model.fit(X_tv, y_tv, verbose=False)
    return model


def plot_score_distribution(df, target, out_path):
    vals = df[target].dropna().values
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=np.arange(0.5, 10.6, 0.5))
    ax.set_title(f"{target} distribution")
    ax.set_xlabel("Human rating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_perturbation_trend(trend_df, task, out_path):
    sub = trend_df[trend_df["task"] == task].copy()
    sub["perturbation"] = pd.Categorical(sub["perturbation"], PERT_ORDER, ordered=True)
    sub = sub.sort_values("perturbation")
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(sub["perturbation"].astype(str), sub["mean_score"], marker="o")
    ax.set_title(f"{task}: rating vs perturbation")
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("Mean human rating")
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(metrics, task, out_path):
    sub = metrics[metrics["task"] == task].copy()
    order = ["Expert-XGB", "ViT-XGB", "Fusion-XGB"]
    sub["model"] = pd.Categorical(sub["model"], order, ordered=True)
    sub = sub.sort_values("model")
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(sub["model"].astype(str), sub["RMSE"])
    ax.set_title(f"{task}: Test RMSE")
    ax.set_ylabel("RMSE (lower is better)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_expert_importance(importance_df, task, out_path):
    sub = importance_df[
        (importance_df["task"] == task)
        & (importance_df["model"] == "Fusion-XGB")
        & (importance_df["feature_group"] == "professional")
    ].nlargest(12, "gain")
    sub = sub.sort_values("gain")
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111)
    ax.barh(sub["feature"], sub["gain"])
    ax.set_title(f"{task}: Professional-feature gain importance")
    ax.set_xlabel("XGBoost gain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    seed_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)

    df = build_dataset()

    # Data statistics and plots.
    stats_rows = []
    for target in ["alignment_score", "overlap_score", "whitespace_score"]:
        task = target.replace("_score", "")
        for split in ["all", "train", "val", "test"]:
            vals = df[target].dropna() if split == "all" else df.loc[df["split"] == split, target].dropna()
            stats_rows.append({
                "task": task,
                "split": split,
                "n": int(len(vals)),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "median": float(vals.median()),
            })
        plot_score_distribution(df, target, OUT / "plots" / f"{task}_score_distribution.png")
    pd.DataFrame(stats_rows).to_csv(OUT / "data_statistics.csv", index=False, encoding="utf-8-sig")

    trend_rows = []
    for target in ["alignment_score", "overlap_score", "whitespace_score"]:
        task = target.replace("_score", "")
        for pert in PERT_ORDER:
            vals = df.loc[(df["perturbation"] == pert) & df[target].notna(), target]
            trend_rows.append({
                "task": task,
                "perturbation": pert,
                "n": int(len(vals)),
                "mean_score": float(vals.mean()),
                "std_score": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
            })
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(OUT / "perturbation_trends.csv", index=False, encoding="utf-8-sig")
    for task in CONFIGS:
        plot_perturbation_trend(trend_df, task, OUT / "plots" / f"{task}_perturbation_trend.png")

    # Expert features.
    expert = extract_professional(df)
    df_feat = df.merge(expert, on="sample_id", how="left", validate="one_to_one")
    df_feat.to_csv(OUT / "professional_features.csv", index=False, encoding="utf-8-sig")

    # ViT features.
    ids, vit = extract_vit(df_feat, batch_size=16)
    if ids != df_feat["sample_id"].tolist():
        raise RuntimeError("ViT feature order mismatch")
    np.save(OUT / "vit_features_768.npy", vit)
    pd.DataFrame({"sample_id": ids, "row_index": np.arange(len(ids))}).to_csv(
        OUT / "vit_feature_index.csv", index=False, encoding="utf-8-sig"
    )

    # Fit scaler + PCA using all training images only (unsupervised, no val/test leakage).
    train_mask_all = df_feat["split"].eq("train").values
    scaler = StandardScaler()
    scaler.fit(vit[train_mask_all])
    vit_scaled_train = scaler.transform(vit[train_mask_all])
    n_pc = min(64, vit_scaled_train.shape[0]-1, vit_scaled_train.shape[1])
    pca = PCA(n_components=n_pc, random_state=SEED)
    pca.fit(vit_scaled_train)
    vit_pca = pca.transform(scaler.transform(vit)).astype(np.float32)
    joblib.dump(scaler, OUT / "models" / "vit_scaler.joblib")
    joblib.dump(pca, OUT / "models" / "vit_pca64.joblib")
    np.save(OUT / "vit_features_pca64.npy", vit_pca)
    pd.DataFrame(
        {
            "component": [f"vit_pc_{i+1:03d}" for i in range(n_pc)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(OUT / "vit_pca_variance.csv", index=False, encoding="utf-8-sig")

    expert_matrix = df_feat[EXPERT_FEATURES].to_numpy(dtype=np.float32)
    pc_names = [f"vit_pc_{i+1:03d}" for i in range(n_pc)]

    metrics_rows = []
    pred_outputs = {}
    importance_rows = []
    best_params_all = {}

    feature_sets = {
        "Expert-XGB": (expert_matrix, EXPERT_FEATURES),
        "ViT-XGB": (vit_pca, pc_names),
        "Fusion-XGB": (np.hstack([vit_pca, expert_matrix]), pc_names + EXPERT_FEATURES),
    }

    for target in ["alignment_score", "overlap_score", "whitespace_score"]:
        task = target.replace("_score", "")
        print(f"[model] task={task}")
        labeled = df_feat[target].notna().values
        y_all = df_feat[target].to_numpy(dtype=float)

        task_pred = df_feat.loc[labeled, ["sample_id","image_id","group_id","split","perturbation",target]].copy()
        task_pred = task_pred.rename(columns={target:"y_true"})

        for model_name, (X_all, feature_names) in feature_sets.items():
            m_train = labeled & df_feat["split"].eq("train").values
            m_val = labeled & df_feat["split"].eq("val").values
            m_test = labeled & df_feat["split"].eq("test").values

            X_train, y_train = X_all[m_train], y_all[m_train]
            X_val, y_val = X_all[m_val], y_all[m_val]
            X_test, y_test = X_all[m_test], y_all[m_test]

            best_params, val_rmse = tune_xgb(X_train, y_train, X_val, y_val)
            model = fit_final_xgb(X_train, y_train, X_val, y_val, best_params)
            pred_test = model.predict(X_test)
            met = metric_dict(y_test, pred_test)
            met.update({
                "task": task,
                "model": model_name,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "val_RMSE_selected": float(val_rmse),
            })
            metrics_rows.append(met)
            best_params_all[f"{task}__{model_name}"] = best_params

            # All labeled predictions from final train+val model for convenience.
            all_labeled_pred = model.predict(X_all[labeled])
            task_pred[f"pred_{model_name.replace('-XGB','').lower()}"] = all_labeled_pred

            model_path = OUT / "models" / f"{task}__{model_name.lower().replace('-','_')}.json"
            model.save_model(model_path)

            booster_score = model.get_booster().get_score(importance_type="gain")
            # If ndarray input was used, features are f0, f1...
            for idx, feat_name in enumerate(feature_names):
                gain = float(booster_score.get(f"f{idx}", 0.0))
                importance_rows.append({
                    "task": task,
                    "model": model_name,
                    "feature": feat_name,
                    "feature_group": "professional" if feat_name in EXPERT_FEATURES else "vit_pca",
                    "gain": gain,
                })

        pred_outputs[task] = task_pred
        task_pred.to_csv(OUT / f"predictions_{task}.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(metrics_rows)
    metrics = metrics[
        ["task","model","n_train","n_val","n_test","MAE","RMSE","R2","Pearson","Spearman","val_RMSE_selected"]
    ].sort_values(["task","RMSE"])
    metrics.to_csv(OUT / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    importance_df = pd.DataFrame(importance_rows)
    importance_df.to_csv(OUT / "feature_importance_gain.csv", index=False, encoding="utf-8-sig")

    for task in CONFIGS:
        plot_model_comparison(metrics, task, OUT / "plots" / f"{task}_model_comparison.png")
        plot_expert_importance(importance_df, task, OUT / "plots" / f"{task}_professional_importance.png")

    with open(OUT / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, ensure_ascii=False, indent=2)

    # Integrity / methodology note.
    best_rows = []
    for task in CONFIGS:
        sub = metrics[metrics["task"] == task].sort_values("RMSE")
        for _, r in sub.iterrows():
            best_rows.append(
                f"{task:10s} | {r['model']:10s} | RMSE={r['RMSE']:.4f} | "
                f"MAE={r['MAE']:.4f} | R2={r['R2']:.4f} | "
                f"Pearson={r['Pearson']:.4f} | Spearman={r['Spearman']:.4f}"
            )

    readme = f"""ViT + Visual-Communication Features + XGBoost: First-Round Experiment
===================================================================

Dataset
-------
Source: {REPO_ID}
Unique physical images: 700
Base designs: 100
Human-labeled images per task: 400
Group split: train/val/test = 70/15/15 base designs
Important: all variants of the same base design remain in one split.

Tasks
-----
1. Alignment score regression
2. Overlap score regression
3. White-space score regression

Deep visual representation
--------------------------
Model: torchvision ViT-B/16
Pretraining: ImageNet-1K pretrained weights
Feature: 768-D output after replacing the classification head with Identity
For XGBoost, StandardScaler + PCA are fitted on TRAIN images only and reduced to {n_pc} dimensions.

Professional visual-communication features
------------------------------------------
24 named features covering:
- Color/tonality: brightness, saturation, contrast, colorfulness, hue entropy, warm-color ratio
- Space/density: white ratio, black ratio, edge density, low-texture ratio, occupied bounding-box ratio
- Composition: vertical/horizontal symmetry, saliency center distance, rule-of-thirds distance,
  left-right balance, top-bottom balance, border saliency, quadrant saliency entropy
- Typography proxies: morphology-based text-region ratio/count/aspect

Note: the typography features are image-processing proxies, not OCR semantic labels.

XGBoost models
--------------
Expert-XGB: professional features only
ViT-XGB: PCA-reduced ViT features only
Fusion-XGB: ViT PCA features + professional features

Hyperparameters are chosen on the validation split using validation RMSE.
Final models are retrained on train+validation and evaluated once on the held-out test split.

First-round test results
------------------------
{chr(10).join(best_rows)}

Interpretation note
-------------------
feature_importance_gain.csv contains XGBoost gain importance. This is model importance, not causal
evidence about human aesthetic judgments. SHAP will be run in the later interpretability stage.

Files
-----
metrics_summary.csv
data_statistics.csv
perturbation_trends.csv
professional_features.csv
vit_features_768.npy
vit_feature_index.csv
vit_features_pca64.npy
vit_pca_variance.csv
predictions_alignment.csv
predictions_overlap.csv
predictions_whitespace.csv
feature_importance_gain.csv
best_params.json
models/
plots/
"""
    (OUT / "README_实验说明.txt").write_text(readme, encoding="utf-8")

    summary = {
        "seed": SEED,
        "unique_images": int(len(df_feat)),
        "base_design_groups": int(df_feat["group_id"].nunique()),
        "expert_feature_count": len(EXPERT_FEATURES),
        "vit_raw_dim": int(vit.shape[1]),
        "vit_pca_dim": int(n_pc),
        "vit_pca_cumulative_variance": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUT / "experiment_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== METRICS ===")
    print(metrics.to_string(index=False))
    print("\n[DONE]", OUT.resolve())


if __name__ == "__main__":
    main()
