from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
TICKER = "AAPL"
HORIZON = 5
SEQ_LEN = 30
N_SAMPLES = 1000

OUT = Path("artifact/Stock_AAPL_FirstRound")
DATA_RAW = OUT / "data" / "raw"
DATA_PROC = OUT / "data" / "processed"
MODELS = OUT / "models"
RESULTS = OUT / "results"
PLOTS = OUT / "plots"
SRC = OUT / "src"

TECH_FEATURES = [
    "ret_1d",
    "log_ret_1d",
    "gap_return",
    "intraday_return",
    "ma5_ratio",
    "ma10_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "ema12_ratio",
    "ema26_ratio",
    "macd",
    "macd_signal",
    "macd_hist",
    "momentum5",
    "roc5",
    "roc10",
    "roc20",
    "rsi14",
    "bb_width",
    "bb_percent",
    "volatility5",
    "volatility10",
    "volatility20",
    "atr14",
    "stoch_k",
    "stoch_d",
    "high_low_range",
    "volume_change",
    "volume_ratio5",
    "volume_ratio20",
]

assert len(TECH_FEATURES) == 30


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))


def ensure_dirs():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in [DATA_RAW, DATA_PROC, MODELS, RESULTS, PLOTS, SRC]:
        d.mkdir(parents=True, exist_ok=True)


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return either (Price,Ticker) or (Ticker,Price).
        lev0 = list(map(str, df.columns.get_level_values(0)))
        lev1 = list(map(str, df.columns.get_level_values(1)))
        canonical = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if len(canonical.intersection(lev0)) >= 4:
            df.columns = df.columns.get_level_values(0)
        elif len(canonical.intersection(lev1)) >= 4:
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = ["_".join(map(str, c)).strip() for c in df.columns]
    return df


def download_data():
    source = None
    try:
        df = yf.download(
            TICKER,
            period="6y",
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        df = flatten_yf_columns(df)
        if df is not None and len(df) >= 1100:
            source = "Yahoo Finance via yfinance"
    except Exception as e:
        print("[data] yfinance failed:", repr(e))
        df = pd.DataFrame()

    if source is None:
        print("[data] trying Stooq fallback")
        url = "https://stooq.com/q/d/l/?s=aapl.us&i=d"
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        source = "Stooq fallback"

    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required OHLCV columns: {missing}; got {df.columns.tolist()}")

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < 1150:
        raise RuntimeError(f"Not enough daily bars: {len(df)}")

    raw = df.reset_index().rename(columns={df.index.name or "index": "Date"})
    raw.to_csv(DATA_RAW / "AAPL_raw.csv", index=False, encoding="utf-8-sig")
    print(f"[data] source={source}, rows={len(df)}, range={df.index.min().date()}..{df.index.max().date()}")
    return df, source


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    price = x["Adj Close"].astype(float)
    open_ = x["Open"].astype(float)
    high = x["High"].astype(float)
    low = x["Low"].astype(float)
    volume = x["Volume"].astype(float)

    x["ret_1d"] = price.pct_change()
    x["log_ret_1d"] = np.log(price / price.shift(1))
    x["gap_return"] = open_ / x["Close"].shift(1) - 1.0
    x["intraday_return"] = x["Close"] / open_ - 1.0

    for n in [5, 10, 20, 60]:
        ma = price.rolling(n).mean()
        x[f"ma{n}_ratio"] = price / ma - 1.0

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd_abs = ema12 - ema26
    macd_signal_abs = macd_abs.ewm(span=9, adjust=False).mean()
    x["ema12_ratio"] = price / ema12 - 1.0
    x["ema26_ratio"] = price / ema26 - 1.0
    x["macd"] = macd_abs / price
    x["macd_signal"] = macd_signal_abs / price
    x["macd_hist"] = (macd_abs - macd_signal_abs) / price

    x["momentum5"] = price / price.shift(5) - 1.0
    x["roc5"] = price.pct_change(5)
    x["roc10"] = price.pct_change(10)
    x["roc20"] = price.pct_change(20)

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))

    ma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    x["bb_width"] = (upper - lower) / ma20
    x["bb_percent"] = (price - lower) / (upper - lower).replace(0, np.nan)

    for n in [5, 10, 20]:
        x[f"volatility{n}"] = x["log_ret_1d"].rolling(n).std()

    prev_close = x["Close"].shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr14"] = tr.rolling(14).mean() / price

    ll14 = low.rolling(14).min()
    hh14 = high.rolling(14).max()
    x["stoch_k"] = 100 * (x["Close"] - ll14) / (hh14 - ll14).replace(0, np.nan)
    x["stoch_d"] = x["stoch_k"].rolling(3).mean()

    x["high_low_range"] = (high - low) / x["Close"]
    x["volume_change"] = volume.pct_change()
    x["volume_ratio5"] = volume / volume.rolling(5).mean()
    x["volume_ratio20"] = volume / volume.rolling(20).mean()

    x["target_price_5d"] = price.shift(-HORIZON)
    x["future_5d_return_pct"] = 100.0 * np.log(x["target_price_5d"] / price)

    # Replace infs produced by rare zero denominators.
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def build_samples(feat_df: pd.DataFrame):
    valid = feat_df.dropna(subset=TECH_FEATURES + ["future_5d_return_pct", "target_price_5d"]).copy()

    endpoints = list(range(SEQ_LEN - 1, len(valid)))
    if len(endpoints) < N_SAMPLES:
        raise RuntimeError(f"Need {N_SAMPLES} valid sequence endpoints, found {len(endpoints)}")
    endpoints = endpoints[-N_SAMPLES:]

    rows = []
    for sample_no, j in enumerate(endpoints):
        r = valid.iloc[j]
        rows.append(
            {
                "sample_no": sample_no,
                "date": valid.index[j],
                "target_date": valid.index[j] + pd.offsets.BDay(HORIZON),
                "current_price": float(r["Adj Close"]),
                "target_price": float(r["target_price_5d"]),
                "future_5d_return_pct": float(r["future_5d_return_pct"]),
                "valid_endpoint_index": int(j),
            }
        )

    sample_df = pd.DataFrame(rows)

    # Chronological split with a five-sample embargo after train and after validation.
    # 0..694 train (695); 695..699 embargo; 700..844 val (145);
    # 845..849 embargo; 850..999 test (150).
    split = np.array(["embargo"] * N_SAMPLES, dtype=object)
    split[0:695] = "train"
    split[700:845] = "val"
    split[850:1000] = "test"
    sample_df["split"] = split

    if sample_df["date"].duplicated().any():
        raise RuntimeError("Duplicate sample dates detected")

    # Fix target_date to the actual trading date exactly HORIZON rows ahead in the original dataframe.
    date_to_pos = {d: i for i, d in enumerate(feat_df.index)}
    actual_target_dates = []
    for d in sample_df["date"]:
        p = date_to_pos[d]
        actual_target_dates.append(feat_df.index[p + HORIZON])
    sample_df["target_date"] = actual_target_dates

    valid.reset_index().rename(columns={valid.index.name or "index": "Date"}).to_csv(
        DATA_PROC / "AAPL_features.csv", index=False, encoding="utf-8-sig"
    )
    sample_df.to_csv(DATA_PROC / "samples_all_1000.csv", index=False, encoding="utf-8-sig")
    for s in ["train", "val", "test", "embargo"]:
        sample_df[sample_df["split"] == s].to_csv(
            DATA_PROC / f"{s}.csv", index=False, encoding="utf-8-sig"
        )

    print("[samples]", sample_df["split"].value_counts().to_dict())
    return valid, sample_df


def make_sequence_arrays(valid: pd.DataFrame, sample_df: pd.DataFrame, scaler: StandardScaler):
    tech = valid[TECH_FEATURES].to_numpy(np.float32)
    scaled = scaler.transform(tech).astype(np.float32)

    X_seq = np.empty((len(sample_df), SEQ_LEN, len(TECH_FEATURES)), dtype=np.float32)
    X_now = np.empty((len(sample_df), len(TECH_FEATURES)), dtype=np.float32)
    y = sample_df["future_5d_return_pct"].to_numpy(np.float32)

    for i, r in sample_df.iterrows():
        j = int(r["valid_endpoint_index"])
        X_seq[i] = scaled[j - SEQ_LEN + 1 : j + 1]
        X_now[i] = tech[j]

    return X_seq, X_now, y


def fit_sequence_scaler(valid: pd.DataFrame, sample_df: pd.DataFrame, allowed_splits):
    mask = sample_df["split"].isin(allowed_splits)
    positions = sample_df.loc[mask, "valid_endpoint_index"].astype(int).to_list()
    row_ids = set()
    for j in positions:
        row_ids.update(range(j - SEQ_LEN + 1, j + 1))
    rows = sorted(row_ids)
    scaler = StandardScaler()
    scaler.fit(valid.iloc[rows][TECH_FEATURES].to_numpy(np.float32))
    return scaler


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
        )
        self.dropout = nn.Dropout(0.20)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, return_hidden=False):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        pred = self.head(self.dropout(h)).squeeze(-1)
        if return_hidden:
            return pred, h
        return pred


def train_lstm_with_validation(X_train, y_train, X_val, y_val, max_epochs=100, patience=15):
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_s))
    val_x = torch.tensor(X_val)
    val_y = torch.tensor(y_val_s)

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=generator)

    model = LSTMRegressor(X_train.shape[-1], 64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = 1
    wait = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach()))

        model.eval()
        with torch.inference_mode():
            val_loss = float(criterion(model(val_x), val_y).detach())

        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[lstm-stageA] epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, y_scaler, pd.DataFrame(history), best_epoch, best_val


def train_lstm_fixed_epochs(X, y, epochs):
    y_scaler = StandardScaler()
    y_s = y_scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    ds = TensorDataset(torch.tensor(X), torch.tensor(y_s))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=64, shuffle=True, generator=generator)

    model = LSTMRegressor(X.shape[-1], 64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(f"[lstm-final] epoch={epoch:03d} train={hist[-1]['train_loss']:.5f}")
    return model, y_scaler, pd.DataFrame(hist)


def lstm_predict(model, y_scaler, X, batch=128):
    model.eval()
    preds = []
    hiddens = []
    with torch.inference_mode():
        for start in range(0, len(X), batch):
            xb = torch.tensor(X[start:start+batch])
            p, h = model(xb, return_hidden=True)
            preds.append(p.numpy())
            hiddens.append(h.numpy())
    pred_s = np.concatenate(preds)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    hidden = np.vstack(hiddens).astype(np.float32)
    return pred, hidden


def tune_xgb(X_train, y_train, X_val, y_val):
    candidates = []
    for depth in [2, 3, 4]:
        for lr in [0.03, 0.07]:
            for n_est in [150, 300]:
                candidates.append(
                    {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        "min_child_weight": 3,
                        "subsample": 0.85,
                        "colsample_bytree": 0.85,
                        "reg_alpha": 0.05,
                        "reg_lambda": 1.0,
                    }
                )

    best_p = None
    best_rmse = float("inf")
    for p in candidates:
        m = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=2,
            tree_method="hist",
            **p,
        )
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_p = p
    return best_p, best_rmse


def fit_xgb(X, y, params):
    m = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=2,
        tree_method="hist",
        **params,
    )
    m.fit(X, y)
    return m


def safe_corr(fn, y, pred):
    try:
        if np.std(pred) < 1e-12:
            return np.nan
        return float(fn(y, pred).statistic)
    except Exception:
        return np.nan


def evaluate(name, y, pred_return, current_price, target_price):
    pred_price = current_price * np.exp(pred_return / 100.0)
    return {
        "model": name,
        "n_test": int(len(y)),
        "MAE_return_pct": float(mean_absolute_error(y, pred_return)),
        "RMSE_return_pct": float(np.sqrt(mean_squared_error(y, pred_return))),
        "R2_return": float(r2_score(y, pred_return)),
        "Pearson": safe_corr(pearsonr, y, pred_return),
        "Spearman": safe_corr(spearmanr, y, pred_return),
        "Direction_Accuracy": float(np.mean(np.sign(y) == np.sign(pred_return))),
        "Price_MAE": float(mean_absolute_error(target_price, pred_price)),
        "Price_RMSE": float(np.sqrt(mean_squared_error(target_price, pred_price))),
    }, pred_price


def save_plots(raw, history, metrics, pred_df, xgb_importance):
    plt.figure(figsize=(10, 4.8))
    plt.plot(raw.index, raw["Adj Close"])
    plt.title("AAPL Adjusted Close")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close")
    plt.tight_layout()
    plt.savefig(PLOTS / "01_price_history.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(pred_df["actual_return_pct"], bins=30)
    plt.title("Test 5-day log-return distribution")
    plt.xlabel("5-day log return (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS / "02_test_return_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    plt.plot(history["epoch"], history["val_loss"], label="Validation")
    plt.title("LSTM Stage-A loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE on standardized target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "03_lstm_loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(pred_df["target_date"]), pred_df["actual_price"], label="Actual")
    for col, label in [
        ("pred_price_naive", "Naive"),
        ("pred_price_xgboost", "XGBoost"),
        ("pred_price_lstm", "LSTM"),
        ("pred_price_fusion", "LSTM-XGBoost"),
    ]:
        plt.plot(pd.to_datetime(pred_df["target_date"]), pred_df[col], label=label, alpha=0.85)
    plt.title("AAPL 5-day-ahead price reconstruction on test period")
    plt.xlabel("Target date")
    plt.ylabel("Adjusted price")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(PLOTS / "04_test_price_predictions.png", dpi=180)
    plt.close()

    order = ["Naive-ZeroReturn", "RandomForest", "XGBoost", "LSTM", "LSTM-XGBoost"]
    m = metrics.set_index("model").loc[order]
    plt.figure(figsize=(8, 4.8))
    plt.bar(m.index, m["RMSE_return_pct"])
    plt.title("Test RMSE of 5-day return")
    plt.ylabel("RMSE (percentage points)")
    plt.xticks(rotation=18)
    plt.tight_layout()
    plt.savefig(PLOTS / "05_model_comparison_rmse.png", dpi=180)
    plt.close()

    top = xgb_importance.sort_values("gain", ascending=False).head(15).sort_values("gain")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["gain"])
    plt.title("XGBoost technical-feature gain importance")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(PLOTS / "06_xgboost_technical_importance.png", dpi=180)
    plt.close()


def main():
    seed_everything()
    ensure_dirs()
    raw, source = download_data()

    feat = compute_technical_features(raw)
    valid, sample_df = build_samples(feat)

    # Stage A scaler: fit only on training-era sequence rows.
    x_scaler_a = fit_sequence_scaler(valid, sample_df, ["train"])
    X_seq_a, X_now_raw, y = make_sequence_arrays(valid, sample_df, x_scaler_a)

    idx_train = np.where(sample_df["split"].to_numpy() == "train")[0]
    idx_val = np.where(sample_df["split"].to_numpy() == "val")[0]
    idx_test = np.where(sample_df["split"].to_numpy() == "test")[0]
    idx_tv = np.concatenate([idx_train, idx_val])

    # LSTM stage A: choose epoch count without touching test.
    model_a, y_scaler_a, hist_a, best_epoch, best_val_loss = train_lstm_with_validation(
        X_seq_a[idx_train], y[idx_train], X_seq_a[idx_val], y[idx_val]
    )
    hist_a.to_csv(RESULTS / "lstm_stageA_history.csv", index=False, encoding="utf-8-sig")
    torch.save(model_a.state_dict(), MODELS / "lstm_stageA_best.pt")
    joblib.dump(x_scaler_a, MODELS / "sequence_scaler_stageA.joblib")
    joblib.dump(y_scaler_a, MODELS / "target_scaler_stageA.joblib")

    pred_a_all, hidden_a_all = lstm_predict(model_a, y_scaler_a, X_seq_a)

    # Tune tree models on train/validation only.
    xgb_params_tech, val_rmse_tech = tune_xgb(
        X_now_raw[idx_train], y[idx_train], X_now_raw[idx_val], y[idx_val]
    )
    fusion_a = np.hstack([hidden_a_all, X_now_raw])
    xgb_params_fusion, val_rmse_fusion = tune_xgb(
        fusion_a[idx_train], y[idx_train], fusion_a[idx_val], y[idx_val]
    )

    # Stage B final LSTM: train+val using the selected epoch count.
    x_scaler_final = fit_sequence_scaler(valid, sample_df, ["train", "val"])
    X_seq_final, X_now_raw2, y2 = make_sequence_arrays(valid, sample_df, x_scaler_final)
    assert np.allclose(y, y2)
    assert np.allclose(X_now_raw, X_now_raw2)

    model_final, y_scaler_final, hist_final = train_lstm_fixed_epochs(
        X_seq_final[idx_tv], y[idx_tv], best_epoch
    )
    torch.save(model_final.state_dict(), MODELS / "lstm_final.pt")
    joblib.dump(x_scaler_final, MODELS / "sequence_scaler_final.joblib")
    joblib.dump(y_scaler_final, MODELS / "target_scaler_final.joblib")
    hist_final.to_csv(RESULTS / "lstm_final_history.csv", index=False, encoding="utf-8-sig")

    pred_lstm_all, hidden_final_all = lstm_predict(model_final, y_scaler_final, X_seq_final)

    # Final XGBoost and RF models on train+validation.
    xgb_tech = fit_xgb(X_now_raw[idx_tv], y[idx_tv], xgb_params_tech)
    fusion_final = np.hstack([hidden_final_all, X_now_raw])
    xgb_fusion = fit_xgb(fusion_final[idx_tv], y[idx_tv], xgb_params_fusion)

    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=SEED,
        n_jobs=2,
    )
    rf.fit(X_now_raw[idx_tv], y[idx_tv])

    xgb_tech.save_model(MODELS / "xgboost_technical.json")
    xgb_fusion.save_model(MODELS / "xgboost_fusion.json")
    joblib.dump(rf, MODELS / "random_forest.joblib")

    # Test predictions.
    y_test = y[idx_test]
    current_price = sample_df.loc[idx_test, "current_price"].to_numpy(float)
    target_price = sample_df.loc[idx_test, "target_price"].to_numpy(float)

    pred_naive = np.zeros_like(y_test)
    pred_rf = rf.predict(X_now_raw[idx_test])
    pred_xgb = xgb_tech.predict(X_now_raw[idx_test])
    pred_lstm = pred_lstm_all[idx_test]
    pred_fusion = xgb_fusion.predict(fusion_final[idx_test])

    metric_rows = []
    prices = {}
    for name, pred in [
        ("Naive-ZeroReturn", pred_naive),
        ("RandomForest", pred_rf),
        ("XGBoost", pred_xgb),
        ("LSTM", pred_lstm),
        ("LSTM-XGBoost", pred_fusion),
    ]:
        met, pp = evaluate(name, y_test, pred, current_price, target_price)
        metric_rows.append(met)
        prices[name] = pp

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "metrics_test.csv", index=False, encoding="utf-8-sig")

    test_rows = sample_df.iloc[idx_test].copy().reset_index(drop=True)
    pred_df = pd.DataFrame(
        {
            "date": test_rows["date"],
            "target_date": test_rows["target_date"],
            "current_price": current_price,
            "actual_price": target_price,
            "actual_return_pct": y_test,
            "pred_return_naive": pred_naive,
            "pred_return_randomforest": pred_rf,
            "pred_return_xgboost": pred_xgb,
            "pred_return_lstm": pred_lstm,
            "pred_return_fusion": pred_fusion,
            "pred_price_naive": prices["Naive-ZeroReturn"],
            "pred_price_randomforest": prices["RandomForest"],
            "pred_price_xgboost": prices["XGBoost"],
            "pred_price_lstm": prices["LSTM"],
            "pred_price_fusion": prices["LSTM-XGBoost"],
        }
    )
    pred_df.to_csv(RESULTS / "predictions_test.csv", index=False, encoding="utf-8-sig")

    # XGBoost technical-feature importance.
    score = xgb_tech.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame(
        {
            "feature": TECH_FEATURES,
            "gain": [float(score.get(f"f{i}", 0.0)) for i in range(len(TECH_FEATURES))],
        }
    ).sort_values("gain", ascending=False)
    imp.to_csv(RESULTS / "xgboost_technical_feature_importance.csv", index=False, encoding="utf-8-sig")

    # Fusion importance + group aggregation.
    fusion_names = [f"lstm_h_{i+1:02d}" for i in range(64)] + TECH_FEATURES
    fscore = xgb_fusion.get_booster().get_score(importance_type="gain")
    fimp = pd.DataFrame(
        {
            "feature": fusion_names,
            "feature_group": ["LSTM_hidden"] * 64 + ["Technical"] * len(TECH_FEATURES),
            "gain": [float(fscore.get(f"f{i}", 0.0)) for i in range(len(fusion_names))],
        }
    ).sort_values("gain", ascending=False)
    fimp.to_csv(RESULTS / "fusion_feature_importance.csv", index=False, encoding="utf-8-sig")
    fgroup = fimp.groupby("feature_group", as_index=False)["gain"].sum()
    fgroup["gain_share"] = fgroup["gain"] / fgroup["gain"].sum()
    fgroup.to_csv(RESULTS / "fusion_feature_group_importance.csv", index=False, encoding="utf-8-sig")

    # Save extracted deep features.
    deep_df = pd.DataFrame(hidden_final_all, columns=[f"lstm_h_{i+1:02d}" for i in range(64)])
    deep_df.insert(0, "date", sample_df["date"].astype(str).to_numpy())
    deep_df.insert(1, "split", sample_df["split"].to_numpy())
    deep_df.to_csv(RESULTS / "lstm_hidden_features_64.csv", index=False, encoding="utf-8-sig")

    # Summary of feature distributions by split.
    stats = []
    for s in ["train", "val", "test"]:
        ids = np.where(sample_df["split"].to_numpy() == s)[0]
        vals = y[ids]
        stats.append(
            {
                "split": s,
                "n": len(ids),
                "start_date": str(sample_df.iloc[ids]["date"].min().date()),
                "end_date": str(sample_df.iloc[ids]["date"].max().date()),
                "target_mean_pct": float(vals.mean()),
                "target_std_pct": float(vals.std(ddof=1)),
                "up_ratio": float((vals > 0).mean()),
            }
        )
    pd.DataFrame(stats).to_csv(RESULTS / "split_statistics.csv", index=False, encoding="utf-8-sig")

    # Plots.
    save_plots(raw, hist_a, metrics, pred_df, imp)

    # Experiment config and reproducibility files.
    best_by_rmse = metrics.sort_values("RMSE_return_pct").iloc[0].to_dict()
    summary = {
        "ticker": TICKER,
        "data_source": source,
        "raw_rows": int(len(raw)),
        "raw_start": str(raw.index.min().date()),
        "raw_end": str(raw.index.max().date()),
        "candidate_samples": N_SAMPLES,
        "sequence_length": SEQ_LEN,
        "forecast_horizon_trading_days": HORIZON,
        "technical_feature_count": len(TECH_FEATURES),
        "lstm_hidden_dim": 64,
        "splits": {
            "train": int(len(idx_train)),
            "validation": int(len(idx_val)),
            "test": int(len(idx_test)),
            "embargo": int((sample_df["split"] == "embargo").sum()),
        },
        "best_lstm_epoch_selected_on_validation": int(best_epoch),
        "best_lstm_val_loss_standardized": float(best_val_loss),
        "xgb_technical_selected_validation_rmse_pct": float(val_rmse_tech),
        "xgb_fusion_selected_validation_rmse_pct": float(val_rmse_fusion),
        "best_test_model_by_rmse": best_by_rmse,
        "technical_features": TECH_FEATURES,
        "xgb_technical_params": xgb_params_tech,
        "xgb_fusion_params": xgb_params_fusion,
        "seed": SEED,
    }
    with open(OUT / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    req = """yfinance
pandas
numpy
scipy
scikit-learn
torch
xgboost
matplotlib
joblib
"""
    (OUT / "requirements.txt").write_text(req, encoding="utf-8")

    # Include the exact script used.
    shutil.copy2(Path(__file__), SRC / Path(__file__).name)

    readme = f"""AAPL 深度学习 + 机器学习股价预测：第一轮真实实验
================================================

研究目标
--------
使用过去30个交易日的信息预测未来5个交易日的对数收益率，再将预测收益率恢复为未来价格。
主模型为 LSTM 深度时序表征 + XGBoost 技术因子融合。

数据
----
股票: AAPL
来源: {source}
原始日线范围: {raw.index.min().date()} 至 {raw.index.max().date()}
候选预测样本: {N_SAMPLES}
输入窗口: {SEQ_LEN} 个交易日
预测周期: {HORIZON} 个交易日
标签: 100 * ln(P[t+5] / P[t])

时间划分
--------
train: 695
embargo: 5
validation: 145
embargo: 5
test: 150

数据严格按时间排序，不进行随机打乱。
标准化器只在当前训练阶段允许的数据上拟合。
测试集从未参与超参数选择。

30个技术指标
------------
{", ".join(TECH_FEATURES)}

LSTM
----
2层LSTM，hidden size=64，dropout=0.2。
第一阶段仅使用 train 训练，并以 validation 选择训练轮数。
第二阶段使用 train+validation 按选定轮数重新训练最终LSTM，
提取64维隐藏状态作为深度时序特征。

机器学习
--------
RandomForest: 技术指标基线
XGBoost: 30维技术指标
LSTM-XGBoost: 64维LSTM隐藏状态 + 30维技术指标

评价指标
--------
5日收益率: MAE、RMSE、R2、Pearson、Spearman、方向准确率
恢复价格: Price MAE、Price RMSE

第一轮测试集结果
----------------
{metrics.to_string(index=False)}

说明
----
1. 这里的“股价预测”不是直接拟合绝对价格，而是预测未来5日对数收益率，
   再通过当前价格恢复未来价格。
2. 方向准确率只表示本测试集上的符号判断，不代表可交易收益。
3. XGBoost gain 重要性是模型重要性，不是经济因果关系。
4. 结果仅用于教学与研究，不构成投资建议。

主要输出
--------
data/raw/AAPL_raw.csv
data/processed/AAPL_features.csv
data/processed/samples_all_1000.csv
data/processed/train.csv / val.csv / test.csv / embargo.csv
models/lstm_final.pt
models/xgboost_technical.json
models/xgboost_fusion.json
models/random_forest.joblib
results/metrics_test.csv
results/predictions_test.csv
results/lstm_hidden_features_64.csv
results/xgboost_technical_feature_importance.csv
results/fusion_feature_importance.csv
plots/*.png
"""
    (OUT / "README_实验说明.txt").write_text(readme, encoding="utf-8")

    print("\n=== TEST METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[DONE]", OUT.resolve())


if __name__ == "__main__":
    main()
