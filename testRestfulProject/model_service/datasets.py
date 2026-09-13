# -*- coding: utf-8 -*-
"""数据集：**显式**类别映射 + 安全的滑窗取样。

为什么不让服务层直接复用 1DCNN/preprocessing.py：

1. 原 `add_labels()` 的类别号来自 `os.listdir()` 的返回顺序——顺序一变，
   `classification_report` 里的 0..9 就不再对应同一类故障，训练出的模型
   也无法解释。本模块用一张写死的 `CWRU_0HP_CLASSES` 表固定映射，并把
   这张表随模型产物一起存进 meta.json。
2. 原切片在数组尾部越界时，numpy 会返回**更短甚至空**的数组，随后被
   `safe_vstack` 用 NaN 补到固定长度。实测 0HP 里 IR014 文件只有 63788 点，
   而 `number=600, stride=150, length=784` 要求切到 91418 点，
   结果验证集/测试集各有 10% 的样本是整行 NaN。本模块 `strict=True` 时
   直接跳过越界窗口，并把跳过数量回报给调用方（写进训练响应与库表备注）。

注意：原有脚本 `1DCNN/preprocessing.py`、`cwt_cnn/preprocess.py` **未被修改**，
命令行跑它们的行为与之前完全一致；这里是服务侧新增的一条更严格的数据通路。
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# CWRU 0HP 的 10 类：顺序沿用本机 os.listdir 的字母序（= 既有实验结果里的编号），
# 但从此写死在这里，不再依赖文件系统返回顺序。
# ----------------------------------------------------------------------------
CWRU_0HP_CLASSES: list[tuple[str, int, str]] = [
    ("48k_Drive_End_B007_0_122.mat",   0, "滚动体故障-0.007in"),
    ("48k_Drive_End_B014_0_189.mat",   1, "滚动体故障-0.014in"),
    ("48k_Drive_End_B021_0_226.mat",   2, "滚动体故障-0.021in"),
    ("48k_Drive_End_IR007_0_109.mat",  3, "内圈故障-0.007in"),
    ("48k_Drive_End_IR014_0_174.mat",  4, "内圈故障-0.014in"),
    ("48k_Drive_End_IR021_0_213.mat",  5, "内圈故障-0.021in"),
    ("48k_Drive_End_OR007@6_0_135.mat", 6, "外圈故障-0.007in@6点钟"),
    ("48k_Drive_End_OR014@6_0_201.mat", 7, "外圈故障-0.014in@6点钟"),
    ("48k_Drive_End_OR021@6_0_238.mat", 8, "外圈故障-0.021in@6点钟"),
    ("normal_0_97.mat",                9, "正常"),
]

# 体检结果缓存：{目录: (时间戳, 结果)}。数据文件不会一秒一变，进页面时不必反复体检。
_DESCRIBE_CACHE: dict[str, tuple[float, dict]] = {}

_FAULT_PATTERNS = [    (re.compile(r"^48k_Drive_End_B0(\d+)_", re.I), "滚动体故障"),
    (re.compile(r"^48k_Drive_End_IR0(\d+)_", re.I), "内圈故障"),
    (re.compile(r"^48k_Drive_End_OR0(\d+)@", re.I), "外圈故障"),
    (re.compile(r"^normal_", re.I), "正常"),
]


def guess_label(filename: str) -> str:
    """给不在登记表里的 .mat 文件兜底起个可读名字。"""
    for pattern, name in _FAULT_PATTERNS:
        m = pattern.match(filename)
        if m:
            if name == "正常":
                return name
            size = int(m.group(1)) / 1000.0
            return f"{name}-{size:.3f}in"
    return Path(filename).stem


def class_table(data_dir: Path | str) -> list[dict]:
    """返回该目录下「文件 → 类别号 → 标签」的对照表，并标出与磁盘不一致的地方。"""
    data_dir = Path(data_dir)
    present = {p.name for p in data_dir.glob("*.mat")} if data_dir.is_dir() else set()
    known = {name for name, _, _ in CWRU_0HP_CLASSES}
    table: list[dict] = []
    for name, class_id, label in CWRU_0HP_CLASSES:
        table.append({"filename": name, "class_id": class_id, "label": label,
                      "on_disk": name in present,
                      "size_bytes": (data_dir / name).stat().st_size if (data_dir / name).is_file() else None})
    for extra in sorted(present - known):
        table.append({"filename": extra, "class_id": None, "label": guess_label(extra),
                      "on_disk": True, "size_bytes": None, "unregistered": True})
    return table


def read_de_channel(file_path: Path) -> np.ndarray:
    """读取 .mat 里的驱动端(DE)振动通道——与原 preprocessing.py 的取数口径一致。"""
    mat = loadmat(str(file_path))
    for key in mat:
        if "DE" in key and not key.startswith("__"):
            return np.asarray(mat[key]).ravel().astype(np.float64)
    raise KeyError(f"{file_path.name} 中找不到含 'DE' 的通道")


def _slice_windows(signal: np.ndarray, number: int, length: int, stride: int,
                   samp_train: int, strict: bool) -> tuple[list[np.ndarray], list[np.ndarray], int, int]:
    """按原脚本的取样口径切训练/测试窗口，并做越界检查。

    返回 (train_windows, test_windows, skipped_train, skipped_test)。

    关于 `strict`：原脚本对越界窗口会切成"短数组"再补 NaN，而本服务下游
    （`finalize_windows` 的 `np.asarray`）**没有补 NaN 这一步** —— 参差数组在
    numpy ≥1.24 会直接抛 `inhomogeneous shape`，也就是 `strict=False` 一用就崩。
    所以这里不再产出短数组：无论 strict 取值都**跳过越界窗口**，
    跳过数量由 `skipped_out_of_range_total` 记录并写进 `Trainings.Remark`。
    """
    train_windows: list[np.ndarray] = []
    test_windows: list[np.ndarray] = []
    skipped_train = skipped_test = 0

    for j in range(samp_train):
        start = j * stride
        if start + length <= signal.size:
            train_windows.append(signal[start:start + length])
        else:
            skipped_train += 1

    base = samp_train * stride + length
    for h in range(number - samp_train):
        start = base + h * stride
        if start + length <= signal.size:
            test_windows.append(signal[start:start + length])
        else:
            skipped_test += 1

    return train_windows, test_windows, skipped_train, skipped_test


def load_windows(dataset_dir: Path | str, length: int, number: int, stride: int,
                 rate: list[float], normal: bool = True, seed: int = 42,
                 strict: bool = True, legacy_scaler: bool = False) -> dict:
    """加载并切分数据集。

    与既有脚本的关键差异（都是刻意为之，且回报在返回值里）：
      * strict=True：跳过越界窗口而不是补 NaN（原脚本会补出整行 NaN 样本）
      * legacy_scaler=False：StandardScaler 只用训练集 fit（原脚本把训练+测试
        拼在一起 fit，属于把测试集统计量泄漏进训练）
      * seed：固定 StratifiedShuffleSplit 的随机种子（原脚本未固定，同参数两次
        跑出过 0.5933 与 0.750 两种测试准确率）
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{dataset_dir}")

    table = [row for row in class_table(dataset_dir) if row["on_disk"]]
    if not table:
        raise FileNotFoundError(f"{dataset_dir} 下没有任何 .mat 文件")

    samp_train = int(number * (1 - (rate[1] + rate[2])))
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    per_class: list[dict] = []

    # 未登记的 .mat（不在 CWRU_0HP_CLASSES 里）以前会**全部**拿到 `len(table)` 这一个 id：
    # 多个文件被并成同一类，而且 labels 长度与 num_classes 对不上。现在各自一个递增 id。
    registered_ids = [row["class_id"] for row in table if row["class_id"] is not None]
    next_extra_id = (max(registered_ids) + 1) if registered_ids else 0

    for row in table:
        signal = read_de_channel(dataset_dir / row["filename"])
        tr, te, skip_tr, skip_te = _slice_windows(signal, number, length, stride, samp_train, strict)
        if row["class_id"] is not None:
            class_id = row["class_id"]
        else:
            class_id = next_extra_id                      # 一个未登记文件 = 一个独立类别
            next_extra_id += 1
        train_x.extend(tr)
        train_y.extend([class_id] * len(tr))
        test_x.extend(te)
        test_y.extend([class_id] * len(te))
        per_class.append({
            "filename": row["filename"], "class_id": class_id, "label": row["label"],
            "samples_in_file": int(signal.size),
            "train_windows": len(tr), "test_windows": len(te),
            "skipped_out_of_range": {"train": skip_tr, "test": skip_te},
            "nan_windows": int(sum(1 for w in tr + te if w.size != length)),
        })

    # 未登记文件在循环里拿到了递增 id（表里仍是 None），排序时按"生效顺序"排：
    # 直接用 r["class_id"] 会拿 None 和 int 比较 → TypeError（原来只要有未登记 .mat 就必崩）
    labels = [row["label"] for row in sorted(table, key=lambda r: (r["class_id"] is None, r["class_id"] or 0))]
    stats = {
        "dataset_dir": str(dataset_dir),
        "dataset_type": "matlab",
        "strict": strict,
        "legacy_scaler": legacy_scaler,
        "seed": seed,
        # 每个 table 行现在都对应一个独立类别（已登记的用登记 id，未登记的递增）
        "num_classes": len(table),
        "nan_windows_total": int(sum(row["nan_windows"] for row in per_class)),
        "skipped_out_of_range_total": int(sum(
            row["skipped_out_of_range"]["train"] + row["skipped_out_of_range"]["test"] for row in per_class)),
        "unregistered_files": [row["filename"] for row in table if row["class_id"] is None],
        "per_class": per_class,
    }
    return finalize_windows(train_x, train_y, test_x, test_y, labels, rate, normal, seed,
                            legacy_scaler, stats)


def finalize_windows(train_x, train_y, test_x, test_y, labels, rate, normal, seed,
                     legacy_scaler, stats: dict) -> dict:
    """两个数据源（.mat 与表格）共用的收尾：标准化 → 划分 → 打乱。

    标准化参数一并返回：它必须跟着模型落盘，推理侧要用同一套均值/方差。
    """
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)

    if normal:
        if legacy_scaler:                    # 复刻旧脚本：训练+测试一起 fit（统计量泄漏）
            scaler = StandardScaler().fit(np.vstack((train_x, test_x)))
        else:
            scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        scaler_stats = {"mean": scaler.mean_, "scale": scaler.scale_, "legacy": legacy_scaler}
    else:
        scaler_stats = None

    test_size = rate[2] / (rate[1] + rate[2])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    test_y_arr = np.asarray(test_y, dtype=np.int32)
    x_valid, x_test, y_valid, y_test = test_x, test_x, test_y_arr, test_y_arr
    for valid_idx, test_idx in splitter.split(test_x, test_y_arr):
        x_valid, x_test = test_x[valid_idx], test_x[test_idx]
        y_valid, y_test = test_y_arr[valid_idx], test_y_arr[test_idx]

    train_y_arr = np.asarray(train_y, dtype=np.int32)
    order = np.random.RandomState(seed).permutation(len(train_x))
    x_train, y_train = train_x[order], train_y_arr[order]

    stats = {**stats,
             "train_total": int(len(x_train)), "valid_total": int(len(x_valid)),
             "test_total": int(len(x_test))}
    return {
        "X_train": x_train, "y_train": y_train,
        "X_valid": x_valid, "y_valid": y_valid,
        "X_test": x_test, "y_test": y_test,
        "labels": labels, "stats": stats,
        "scaler": scaler_stats,
    }


def describe_dataset(dataset_dir: Path | str, max_age: float = 120.0) -> dict:
    """只做体检，不切数据：给 /models 与 /train 的预检用。

    两个性能要点（之前这里是页面跳转慢的头号原因）：
      1. 取每个 .mat 的采样点数**不要 loadmat**——那会把 24 万点的数组整个读进内存，
         10 个文件就是几百 MB 的解析开销；改用 `whosmat`，只读变量名与形状；
      2. 结果缓存 120 秒（数据文件不会一秒一变），避免每次进页面都重新体检。
    """
    from scipy.io import whosmat
    dataset_dir = Path(dataset_dir)
    key = str(dataset_dir)
    cached = _DESCRIBE_CACHE.get(key)
    if cached and time.time() - cached[0] < max_age:
        return cached[1]

    rows = class_table(dataset_dir)
    present = [r for r in rows if r["on_disk"]]
    samples: dict[str, int] = {}
    for row in present:
        try:
            info = whosmat(str(dataset_dir / row["filename"]))
            samples[row["filename"]] = next((int(shape[0]) for name, shape, _cls in info if "DE" in name), 0)
        except Exception as exc:                                  # pragma: no cover - 数据损坏时兜底
            samples[row["filename"]] = -1
            row["error"] = str(exc)
    for row in rows:
        if row["on_disk"]:
            row["samples_in_file"] = samples.get(row["filename"])
    return {
        "dataset_dir": str(dataset_dir),
        "file_count": len(present),
        "classes": len({r["class_id"] for r in present}),
        "files": rows,
        "min_samples_in_file": min([v for v in samples.values() if v > 0], default=0),
        "max_samples_in_file": max(samples.values(), default=0),
    }
