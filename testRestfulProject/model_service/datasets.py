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
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# CWRU 0HP 的 10 类：顺序沿用本机 os.listdir 的字母序（= 既有实验结果里的编号），
# 但从此写死在这里，不再依赖文件系统返回顺序。
# ----------------------------------------------------------------------------
# ⚠️ 这张表必须**写死**，不能改成遍历 `os.listdir()` 得到的顺序：listdir 的返回顺序由文件系统
#    决定（换机器、复制方式、重命名都会变），顺序一变 class_id 就整体漂移，历史实验里的"类别 4"
#    可能指向另一种故障，classification_report 与旧结论全部无法对照。
# ⚠️ 表内行序 = class_id 递增顺序 = labels 列表顺序（页面/报告里中文标签的展示顺序）。
#    新增类别只能往表尾追加；中间插行、重排或改号都会破坏与既有模型产物的对应关系。
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

# 未登记 .mat 的兜底命名规则：文件名前缀 → 故障部位（捕获组里是故障尺寸，单位 0.001in）。
# 只影响"显示成什么名字"，不影响类别号——未登记文件的类别号在 load_windows 里递增分配。
_FAULT_PATTERNS = [    (re.compile(r"^48k_Drive_End_B0(\d+)_", re.I), "滚动体故障"),
    (re.compile(r"^48k_Drive_End_IR0(\d+)_", re.I), "内圈故障"),
    (re.compile(r"^48k_Drive_End_OR0(\d+)@", re.I), "外圈故障"),
    (re.compile(r"^normal_", re.I), "正常"),
]


def guess_label(filename: str) -> str:
    """给不在登记表里的 .mat 文件兜底起个可读名字。"""
    # 只负责"起个能看懂的名字"：命中原脚本的命名风格 → "部位-尺寸in"；没命中 → 直接用文件名。
    # ⚠️ 名字丑一点没关系，但绝不能在这里定类别号——未登记文件的类别号由 load_windows 递增分配。
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
    # 做的是"登记表 ↔ 磁盘实况"的对照：登记了但盘上没有 → on_disk=False（调用方据此提示缺文件）；
    # 盘上有但没登记 → 追到表尾并打 unregistered=True。体检页、训练侧都靠这张表决定用哪些文件。
    data_dir = Path(data_dir)
    present = {p.name for p in data_dir.glob("*.mat")} if data_dir.is_dir() else set()
    known = {name for name, _, _ in CWRU_0HP_CLASSES}
    table: list[dict] = []
    for name, class_id, label in CWRU_0HP_CLASSES:
        table.append({"filename": name, "class_id": class_id, "label": label,
                      "on_disk": name in present,
                      "size_bytes": (data_dir / name).stat().st_size if (data_dir / name).is_file() else None})
    # 未登记文件按文件名排序追加到表尾：排序只为"体检列表看起来稳定"，
    # 它们的类别号在 load_windows 里才分配（见那边的递增 id 处理）
    for extra in sorted(present - known):
        table.append({"filename": extra, "class_id": None, "label": guess_label(extra),
                      "on_disk": True, "size_bytes": None, "unregistered": True})
    return table


def read_de_channel(file_path: Path) -> np.ndarray:
    """读取 .mat 里的驱动端(DE)振动通道——与原 preprocessing.py 的取数口径一致。"""
    # CWRU 一个 .mat 里塞了多个变量（X###_DE_time / _FE_time / _BA_time / RPM…），全项目统一
    # 只用驱动端 DE，保证不同脚本、不同实验喂给模型的是同一条物理通道。
    mat = loadmat(str(file_path))
    # ⚠️ 判定用的是**大写** "DE"（大小写敏感，CWRU 原始变量名就是 X###_DE_time），并跳过 loadmat
    #    自动注入的 __header__/__version__/__globals__ 元信息键。
    # ⚠️ 取"第一个键名含 DE 的变量"；一个都没命中就抛 KeyError，绝不静默退化成"随便取第一个变量"
    #    ——否则风扇端/基座信号会被混进训练集，指标看着正常却无法解释。
    for key in mat:
        if "DE" in key and not key.startswith("__"):
            # ravel()：.mat 里的变量可能是 (N,1) 二维，统一拉成 1D，下游切窗只认一维数组
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

    # 训练窗：从 j*stride 起（j = 0..samp_train-1），每 stride 取一个 length 长的窗。
    for j in range(samp_train):
        start = j * stride
        # ⚠️ 越界窗口整窗丢弃、只计数：既不截短成短数组，也绝不补 NaN。补 NaN 会造出整行 NaN 的
        #    "假样本"（0HP 的 IR014 只有 63788 点，而 number=600/stride=150/length=784 要求切到
        #    91418 点，旧脚本验证集/测试集各有 10% 是 NaN 行）；截短则会让下游 np.asarray 抛
        #    inhomogeneous shape（见 docstring）。
        if start + length <= signal.size:
            train_windows.append(signal[start:start + length])
        else:
            skipped_train += 1

    # 测试窗起点 = samp_train*stride + length：先空出 length 的间隔再往后切，
    # 保证测试段与最后一个训练窗**不重叠**——否则同一段波形既训练又测试，指标会虚高。
    base = samp_train * stride + length
    for h in range(number - samp_train):
        start = base + h * stride
        # 越界处理同训练窗：计数后跳过，两个数据源口径保持一致
        if start + length <= signal.size:
            test_windows.append(signal[start:start + length])
        else:
            skipped_test += 1

    # ⚠️ 入参 strict 现在两种取值行为**完全一致**（都跳过越界窗口），它只为兼容历史调用签名而保留。
    #    历史上 strict=False 会产出短数组，而下游没有补 NaN 这一步，等于"一用就崩"，
    #    所以不要指望 strict=False 能多榨出几个样本。
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

    # 只有"磁盘上真的存在"的行才参与训练：登记表里缺失的文件只出现在体检结果里（on_disk=False）
    table = [row for row in class_table(dataset_dir) if row["on_disk"]]
    if not table:
        raise FileNotFoundError(f"{dataset_dir} 下没有任何 .mat 文件")

    # 训练窗口数由 number 与"验证+测试"占比反推：number*(1-rate[1]-rate[2])。
    # ⚠️ rate[0]（名义上的 train 比例）**在这里不参与任何运算**，详见 finalize_windows 里的说明。
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

    # 逐文件读 DE 通道 → 切窗 → 累积样本。每个文件独立切窗，各类样本数天然不均，
    # 所以 per_class 会逐个文件记真实窗口数与跳过数，方便排查"某一类被切空"。
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
        "num_classes": len(table),        "nan_windows_total": int(sum(row["nan_windows"] for row in per_class)),
        "skipped_out_of_range_total": int(sum(
            row["skipped_out_of_range"]["train"] + row["skipped_out_of_range"]["test"] for row in per_class)),
        # 未登记文件清单回报给调用方（训练响应 / Trainings.Remark）：它们各自占一个递增 id，
        # 所以 labels 长度与 num_classes 始终对齐——以前这些文件全拿 len(table) 同一个 id，
        # 多个文件被悄悄并成同一类，模型"看懂"的类别数与页面上显示的并不一致。
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
    # ⚠️ np.asarray 要求所有窗口长度完全一致：切窗函数必须已经"跳过越界窗口"才行，
    #    一旦混进短数组，numpy ≥1.24 会直接抛 inhomogeneous shape（这一层没有补 NaN 的兜底）。
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)

    # 标准化：默认只用训练集 fit，推理侧必须复用同一套 mean/scale，故 scaler_stats 随模型落盘。
    if normal:
        # ⚠️ legacy_scaler=True 复刻的是旧脚本的**有瑕疵**做法：把 train+test 拼起来 fit，
        #    测试集统计量泄漏进训练（指标偏乐观、上线掉点）。它只为对齐历史数值而留，
        #    新实验一律保持 False。
        if legacy_scaler:                    # 复刻旧脚本：训练+测试一起 fit（统计量泄漏）
            scaler = StandardScaler().fit(np.vstack((train_x, test_x)))
        else:
            scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        scaler_stats = {"mean": scaler.mean_, "scale": scaler.scale_, "legacy": legacy_scaler}
    else:
        scaler_stats = None

    # ⚠️ rate = [train, valid, test] 里**只有 rate[1]、rate[2] 真正参与运算**：训练集规模在上游由
    #    number*(1-rate[1]-rate[2]) 定死，rate[0] 只被写进 meta/stats 供展示；这里的 test_size 是
    #    "测试集在 验证+测试 池里的占比" = rate[2]/(rate[1]+rate[2])，
    #    所以传 0.6/0.2/0.2 与 0.8/0.2/0.2 切出的 训练/验证/测试 规模完全相同。
    test_size = rate[2] / (rate[1] + rate[2])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    test_y_arr = np.asarray(test_y, dtype=np.int32)
    # ⚠️ 命名陷阱：进循环前 test_x/test_y 装的是"验证集+测试集"的合并池（不是最终测试集），
    #    下面的 StratifiedShuffleSplit 才把它切成 x_valid / x_test 两份；
    #    这一行同池兜底赋值只是"划分没有产出任何一项"时的保险，正常路径会被循环体覆盖。
    # ⚠️ 划分是分层的：池子里某一类的样本数 < 2 时 sklearn 会直接报错（"least populated class
    #    has only 1 member"），所以上游必须保证每个类别切出的窗口足够多。
    x_valid, x_test, y_valid, y_test = test_x, test_x, test_y_arr, test_y_arr
    for valid_idx, test_idx in splitter.split(test_x, test_y_arr):
        x_valid, x_test = test_x[valid_idx], test_x[test_idx]
        y_valid, y_test = test_y_arr[valid_idx], test_y_arr[test_idx]

    # 训练集额外洗一次牌并固定随机种子（原脚本没固定，同参数两次跑出过 0.5933 与 0.750 两种测试准确率）；
    # 洗牌是为了打断"同一文件、同一时间段的窗口连续喂入"造成的顺序偏差。
    # ⚠️ 注意只有训练集在这里被打乱：valid/test 是上面分层划分出来的，类内顺序已被 seed 打乱过，
    #    不再保持"按文件、按波形时间"的原始次序——别指望用测试集的行号映射回原文件的位置。
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


def describe_dataset(dataset_dir: Path | str) -> dict:
    """只做体检，不切数据：给 /models 与 /train 的预检用。

    性能要点（之前这里是页面跳转慢的头号原因）：取每个 .mat 的采样点数**不要 loadmat**——
    那会把 24 万点的数组整个读进内存，10 个文件就是几百 MB 的解析开销；改用 `whosmat`，
    只读变量名与形状。
    ⚠️ 这里原先还有一层"120 秒结果缓存"（模块级 `_DESCRIBE_CACHE`，配一个 `max_age` 形参），
    已整体删除：那个字典**只被读、从来没有被写**，缓存永远是空的，注释承诺的 120 秒缓存从未生效，
    只是让读的人以为有缓存可清；而且它的键与淘汰策略还都是一笔糊涂账。改为直接体检后行为不变
    ——`whosmat` 本身已经足够快（见上一段）。
    """
    # whosmat 是"轻量版 loadmat"：只解析 .mat 头信息（变量名 + 形状），不碰数组体。
    # ⚠️ 千万别在这里换成 loadmat：24 万点的数组 × 10 个文件 = 几百 MB 的解析与内存开销，
    #    那正是 /models、/train 预检页面卡顿的头号原因——体检只需要采样点数，不需要数据本身。
    from scipy.io import whosmat
    dataset_dir = Path(dataset_dir)

    rows = class_table(dataset_dir)
    present = [r for r in rows if r["on_disk"]]
    samples: dict[str, int] = {}
    for row in present:
        try:
            # 与 read_de_channel 同口径：只认变量名含 DE 的通道，取形状的第一维当采样点数；
            # 一个都没找到就给 0（页面上显示"0 点"，而不是编一个看起来正常的数字）
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
