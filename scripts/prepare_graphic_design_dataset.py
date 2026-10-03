from pathlib import Path
import hashlib
import json
import math
import random
import shutil

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO_ID = "creative-graphic-design/GraphicDesignEvaluation"
CONFIGS = {
    "alignment": "absolute-human-alignment",
    "overlap": "absolute-human-overlap",
    "whitespace": "absolute-human-whitespace",
}
SEED = 42
OUT = Path("artifact/GraphicDesignEvaluation_ready")
IMG_ROOT = OUT / "images"
RAW_ROOT = OUT / "raw_labels"

def ensure_clean():
    if OUT.exists():
        shutil.rmtree(OUT)
    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

def perturbation_name(ds, value):
    feat = ds.features["perturbation"]
    if hasattr(feat, "int2str") and isinstance(value, (int, np.integer)):
        return feat.int2str(int(value))
    return str(value).strip().lower()

def safe_float(v):
    if v is None:
        return np.nan
    try:
        x = float(v)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan

def save_png(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    rgb.save(path, format="PNG", optimize=True)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    ensure_clean()
    records = {}
    config_stats = {}
    original_hashes = {}

    # Wide columns for five individual raters per principle.
    score_columns = {
        p: [f"{p}_rater_{i+1}" for i in range(5)]
        for p in CONFIGS
    }

    for principle, config in CONFIGS.items():
        print(f"Loading {config} ...")
        ds = load_dataset(REPO_ID, config, split="train")
        if len(ds) != 400:
            raise RuntimeError(f"{config}: expected 400 rows, got {len(ds)}")

        raw_rows = []
        for idx, row in enumerate(ds):
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                raise RuntimeError(f"{config} row {idx}: missing image_id")

            pert = perturbation_name(ds, row["perturbation"])
            if pert not in {"none", "small", "medium", "large"}:
                raise RuntimeError(f"{config} row {idx}: unexpected perturbation={pert}")

            scores = [safe_float(x) for x in list(row["scores"])]
            valid_scores = [x for x in scores if not np.isnan(x)]
            avg = safe_float(row["avg"])
            if not valid_scores:
                raise RuntimeError(f"{config} row {idx}: no valid human ratings")
            calc_avg = float(np.mean(valid_scores))
            if abs(calc_avg - avg) > 1e-3:
                raise RuntimeError(
                    f"{config} row {idx}: avg mismatch, source={avg}, recomputed={calc_avg}"
                )

            if pert == "none":
                sample_id = f"{image_id}__original"
                image_rel = Path("images") / "original" / f"{image_id}.png"
                target_type = "overall_original"
            else:
                sample_id = f"{image_id}__{principle}__{pert}"
                image_rel = Path("images") / principle / pert / f"{image_id}.png"
                target_type = principle

            rec = records.setdefault(sample_id, {
                "sample_id": sample_id,
                "image_id": image_id,
                "group_id": image_id,
                "image_path": image_rel.as_posix(),
                "perturbation": pert,
                "target_type": target_type,
                "alignment_score": np.nan,
                "overlap_score": np.nan,
                "whitespace_score": np.nan,
                **{c: np.nan for cols in score_columns.values() for c in cols},
                "image_sha256": "",
            })

            score_key = f"{principle}_score"
            if not pd.isna(rec[score_key]):
                raise RuntimeError(f"duplicate label for {sample_id}: {principle}")
            rec[score_key] = avg
            for i, val in enumerate(scores[:5]):
                rec[score_columns[principle][i]] = val

            image_path = OUT / image_rel
            if not image_path.exists():
                digest = save_png(row["image"], image_path)
                rec["image_sha256"] = digest
            else:
                # Verify that the three original-image configs truly contain the same pixels.
                tmp = OUT / "_tmp_verify.png"
                digest = save_png(row["image"], tmp)
                with open(image_path, "rb") as f:
                    existing_digest = hashlib.sha256(f.read()).hexdigest()
                tmp.unlink(missing_ok=True)
                if digest != existing_digest:
                    raise RuntimeError(
                        f"image mismatch for shared sample {sample_id} across configs"
                    )
                rec["image_sha256"] = existing_digest

            if pert == "none":
                original_hashes.setdefault(image_id, set()).add(rec["image_sha256"])

            raw_rows.append({
                "image_id": image_id,
                "perturbation": pert,
                "rater_1": scores[0] if len(scores) > 0 else np.nan,
                "rater_2": scores[1] if len(scores) > 1 else np.nan,
                "rater_3": scores[2] if len(scores) > 2 else np.nan,
                "rater_4": scores[3] if len(scores) > 3 else np.nan,
                "rater_5": scores[4] if len(scores) > 4 else np.nan,
                "avg": avg,
            })

        pd.DataFrame(raw_rows).to_csv(
            RAW_ROOT / f"human_abs_{principle}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        config_stats[principle] = {
            "rows": len(ds),
            "unique_image_ids": len(set(x["image_id"] for x in raw_rows)),
            "score_mean": float(pd.DataFrame(raw_rows)["avg"].mean()),
            "score_std": float(pd.DataFrame(raw_rows)["avg"].std()),
            "score_min": float(pd.DataFrame(raw_rows)["avg"].min()),
            "score_max": float(pd.DataFrame(raw_rows)["avg"].max()),
        }

    # Complete derived labels.
    df = pd.DataFrame(records.values())
    expected_groups = df["group_id"].nunique()
    expected_rows = expected_groups * 10
    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected 10 physical images per base design; got {len(df)} rows for "
            f"{expected_groups} groups (expected {expected_rows})"
        )

    def overall(row):
        vals = [row["alignment_score"], row["overlap_score"], row["whitespace_score"]]
        vals = [float(x) for x in vals if not pd.isna(x)]
        return float(np.mean(vals)) if len(vals) == 3 else np.nan

    df["overall_score"] = df.apply(overall, axis=1)
    # target_score is convenient for baseline modeling, but target_type MUST be retained.
    df["target_score"] = np.where(
        df["perturbation"].eq("none"),
        df["overall_score"],
        np.select(
            [
                df["target_type"].eq("alignment"),
                df["target_type"].eq("overlap"),
                df["target_type"].eq("whitespace"),
            ],
            [
                df["alignment_score"],
                df["overlap_score"],
                df["whitespace_score"],
            ],
            default=np.nan,
        ),
    )

    # Deterministic group split: all variants of one source design stay together.
    groups = sorted(df["group_id"].unique().tolist())
    rng = random.Random(SEED)
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train+n_val])
    test_groups = set(groups[n_train+n_val:])

    def split_name(g):
        if g in train_groups:
            return "train"
        if g in val_groups:
            return "val"
        return "test"

    df["split"] = df["group_id"].map(split_name)
    split_order = pd.Categorical(df["split"], ["train", "val", "test"], ordered=True)
    df = df.assign(_split_order=split_order).sort_values(
        ["_split_order", "group_id", "perturbation", "target_type", "sample_id"]
    ).drop(columns="_split_order").reset_index(drop=True)

    # Integrity checks.
    if df["sample_id"].duplicated().any():
        raise RuntimeError("duplicate sample_id detected")
    missing_paths = [p for p in df["image_path"] if not (OUT / p).exists()]
    if missing_paths:
        raise RuntimeError(f"missing image files: {missing_paths[:5]}")
    if any(len(v) != 1 for v in original_hashes.values()):
        raise RuntimeError("shared original images are not identical across principles")

    # Save master and splits.
    df.to_csv(OUT / "labels.csv", index=False, encoding="utf-8-sig")
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_csv(
            OUT / f"{split}.csv", index=False, encoding="utf-8-sig"
        )

    # Label dictionary workbook.
    dictionary = [
        ("sample_id", "样本唯一ID", "原图为 image_id__original；扰动图含评价维度与扰动等级"),
        ("image_id", "原始设计ID", "同一 image_id 的所有版本必须在同一数据划分中"),
        ("group_id", "分组ID", "等于 image_id，用于 Group Split，防止数据泄漏"),
        ("image_path", "图像相对路径", "相对于数据包根目录"),
        ("perturbation", "扰动等级", "none / small / medium / large"),
        ("target_type", "当前样本主要评价目标", "原图为 overall_original；扰动图为 alignment/overlap/whitespace"),
        ("alignment_score", "对齐人工平均分", "1-10；无该标签时为空"),
        ("overlap_score", "重叠人工平均分", "1-10；无该标签时为空"),
        ("whitespace_score", "留白人工平均分", "1-10；无该标签时为空"),
        ("overall_score", "原图三维综合分", "仅原图有值；三项人工平均分的等权平均"),
        ("target_score", "便捷建模目标分", "原图=overall_score；扰动图=对应 target_type 的人工评分"),
        ("*_rater_1...5", "单个标注者分数", "保留官方原始人工评分；个别源记录可能少于5个有效评分"),
        ("image_sha256", "图像SHA256", "用于完整性与重复检查"),
        ("split", "数据划分", "train / val / test；按 group_id 分组切分"),
    ]
    pd.DataFrame(dictionary, columns=["字段", "中文含义", "说明"]).to_excel(
        OUT / "label_description.xlsx", index=False
    )

    # Statistics workbook.
    overall_stats = pd.DataFrame([
        {"指标": "物理图像样本数", "值": len(df)},
        {"指标": "原始设计组数", "值": expected_groups},
        {"指标": "训练集样本数", "值": int((df["split"]=="train").sum())},
        {"指标": "验证集样本数", "值": int((df["split"]=="val").sum())},
        {"指标": "测试集样本数", "值": int((df["split"]=="test").sum())},
        {"指标": "原图样本数", "值": int((df["perturbation"]=="none").sum())},
        {"指标": "扰动图样本数", "值": int((df["perturbation"]!="none").sum())},
    ])
    principle_stats = pd.DataFrame(config_stats).T.reset_index().rename(columns={"index":"principle"})
    split_stats = df.groupby(["split", "target_type"], observed=True).size().reset_index(name="count")
    perturb_stats = df.groupby(["perturbation", "target_type"], observed=True).size().reset_index(name="count")
    with pd.ExcelWriter(OUT / "dataset_statistics.xlsx", engine="openpyxl") as writer:
        overall_stats.to_excel(writer, sheet_name="总体", index=False)
        principle_stats.to_excel(writer, sheet_name="三维人工评分", index=False)
        split_stats.to_excel(writer, sheet_name="数据划分", index=False)
        perturb_stats.to_excel(writer, sheet_name="扰动分布", index=False)

    readme = f"""GraphicDesignEvaluation_ready
=============================

用途
----
视觉传达 / 平面设计质量评价实验。数据只使用官方 Human Absolute Evaluation，
不使用 GPT 自动评分充当人工标签。

来源
----
Hugging Face: creative-graphic-design/GraphicDesignEvaluation
原始项目: CyberAgentAILab/Graphic-design-evaluation
论文: Can GPTs Evaluate Graphic Design Based on Design Principles?
SIGGRAPH Asia 2024 Technical Communications
许可证: Apache-2.0

本整理包
--------
物理图像样本: {len(df)}
原始设计组: {expected_groups}
每个原始设计组包含:
  - 1 张原图
  - alignment 的 small/medium/large 三张扰动图
  - overlap 的 small/medium/large 三张扰动图
  - whitespace 的 small/medium/large 三张扰动图
因此每组 10 张图。

人工标签
--------
alignment_score: 对齐质量评分
overlap_score: 元素重叠质量评分
whitespace_score: 留白质量评分
每项来自人工绝对评价，原始数据通常由多名标注者评分并提供 avg。
本包同时保留 rater_1...5，未擅自生成或替换人工标签。

overall_score
-------------
仅对未扰动原图计算：
overall_score = (alignment_score + overlap_score + whitespace_score) / 3
这是本整理过程生成的“派生变量”，不是官方直接提供的综合评分，因此论文中必须明确说明。

target_score
------------
便于建立第一版模型：
- 原图：取 overall_score
- 扰动图：取当前受扰动设计原则的人工平均分
注意：如果建立单一回归器，应同时保留 target_type；更严谨的做法是分别训练
alignment / overlap / whitespace 三个回归任务，或做多任务学习。

数据划分
--------
随机种子: {SEED}
按 image_id / group_id 进行分组切分，避免同一原设计的不同扰动版本同时出现在
训练集和测试集造成数据泄漏。
train / val / test 约为 70% / 15% / 15% 的原始设计组。

主要文件
--------
labels.csv                 全部整理后的标签
train.csv                  训练集
val.csv                    验证集
test.csv                   测试集
label_description.xlsx     字段中文说明
dataset_statistics.xlsx    数据统计
raw_labels/                三个评价维度的原始人工绝对评价表
images/                    图像文件

建议后续论文实验
--------------
1. 分别以 alignment_score / overlap_score / whitespace_score 为回归标签。
2. ViT 提取深度视觉特征。
3. 提取色彩、构图、留白、文字密度等可解释设计特征。
4. 使用 XGBoost 融合特征并预测人工评分。
5. 使用 SHAP 分析专业设计特征的贡献。
"""
    (OUT / "README_数据说明.txt").write_text(readme, encoding="utf-8")

    manifest = {
        "source_repo": REPO_ID,
        "configs": CONFIGS,
        "seed": SEED,
        "rows": int(len(df)),
        "groups": int(expected_groups),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "sha256_labels_csv": hashlib.sha256((OUT/"labels.csv").read_bytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("DONE:", OUT.resolve())

if __name__ == "__main__":
    main()
