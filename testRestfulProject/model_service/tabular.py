# -*- coding: utf-8 -*-
"""表格数据集（Excel / CSV）：**一个文件 = 一个类别，文件名即标签，表内指定一列作信号**。

这是和 CWRU `.mat` 平级的第二种数据源，两者共用同一套切窗/标准化/划分逻辑
（`datasets.finalize_windows`），因此 1DCNN / cwt_cnn 不需要任何改动就能跑表格数据。

约定与限制：
  * 支持 `.csv` `.txt` `.xlsx` `.xlsm` `.xls`（xlsx 走 openpyxl，xls 走 xlrd）
  * 每个文件一个类别；类别名 = 文件名（去掉扩展名）
  * 信号列：可显式指定 `column=`；否则按「列名像信号」(振幅/振动/value/signal…) →
    「首个数值列」的顺序自动挑；时间/序号列会被优先排除
  * CSV 编码依次尝试 utf-8-sig → utf-8 → gbk（中文 Excel 导出的 CSV 常见 GBK）
  * 越界窗口与 .mat 一样：strict=True 跳过并回报，绝不补 NaN
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import datasets as ds

TABLE_SUFFIXES = {".csv", ".txt", ".xlsx", ".xlsm", ".xls"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
SIGNAL_HINTS = ("振幅", "振动", "幅值", "加速度", "signal", "value", "amplitude", "vibration",
                "de_time", "de", "acc")
TIME_HINTS = ("时间", "时刻", "序号", "采样点序号", "time", "timestamp", "date", "index", "no.")


def is_table(path: Path | str) -> bool:
    """是不是一张可读的表格（只看扩展名，不打开文件）。"""
    return Path(path).suffix.lower() in TABLE_SUFFIXES


def has_tables(directory: Path | str) -> bool:
    """目录里是否存在表格文件——训练侧据此自动判定数据源类型（matlab / tabular）。"""
    directory = Path(directory)
    return directory.is_dir() and any(is_table(p) for p in directory.iterdir() if p.is_file())


def list_table_files(directory: Path | str) -> list[Path]:
    """按文件名排序返回表格文件（排序即类别号顺序，稳定可复现）。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and is_table(p)],
                  key=lambda p: p.name.lower())


def label_from_filename(filename: str) -> str:
    """类别名 = 文件名去扩展名（不做其它改写，保证可追溯）。"""
    return Path(filename).stem


# ------------------------------------------------------------------ 读表
def read_table(path: Path | str, sheet: str | int | None = None) -> pd.DataFrame:
    """读成 DataFrame。Excel 按后缀挑引擎，CSV 依次试编码并在必要时重猜分隔符。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        # .xls 是老格式，只有 xlrd 能读；.xlsx/.xlsm 交给 openpyxl
        engine = "xlrd" if suffix == ".xls" else "openpyxl"
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, engine=engine)
    last_error = None
    # 编码顺序是照中文 Excel 导出的实际情况排的：utf-8-sig（带 BOM）最保险，GBK 兜底
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            if frame.shape[1] == 1:                       # 可能是分号/制表符分隔
                alt = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
                if alt.shape[1] > frame.shape[1]:         # 猜出来的列更多才采纳，避免把单列数据拆坏
                    frame = alt
            return frame
        except UnicodeDecodeError as exc:                  # 换编码重试
            last_error = exc
            continue
        except Exception as exc:                           # 其它错误（空文件、格式坏）换编码也没用
            last_error = exc
            break
    raise ValueError(f"读取失败：{path.name}（{last_error}）")


def list_sheets(path: Path | str) -> list[str]:
    """Excel 的 sheet 名列表（非 Excel 或读失败一律返回空列表，调用方无需处理异常）。"""
    path = Path(path)
    if path.suffix.lower() not in EXCEL_SUFFIXES:
        return []
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    try:
        return list(pd.ExcelFile(path, engine=engine).sheet_names)
    except Exception:
        return []


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """真正的数值列：dtype 是数字，或整列都能转成数字且非全空。"""
    out = []
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_numeric_dtype(series):
            if series.notna().any():
                out.append(str(col))
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() >= max(1, int(0.8 * series.notna().sum())):
            out.append(str(col))
    return out


def pick_signal_column(frame: pd.DataFrame, column: str | None = None) -> str:
    """选信号列：显式指定 > 列名像信号 > 首个数值列。"""
    numeric = numeric_columns(frame)
    if not numeric:
        raise ValueError(f"表里没有可用的数值列；列名：{list(map(str, frame.columns))}")
    if column:
        if column not in map(str, frame.columns):
            raise ValueError(f"列 {column!r} 不存在；可选数值列：{numeric}")
        if column not in numeric:
            raise ValueError(f"列 {column!r} 不是数值列；可选数值列：{numeric}")
        return column

    def named_like_signal(name: str) -> bool:
        """列名看着像信号吗：命中 SIGNAL_HINTS 且没命中 TIME_HINTS（时间列优先排除）。"""
        low = name.lower()
        if any(hint in low for hint in TIME_HINTS):
            return False
        return any(hint in low for hint in SIGNAL_HINTS)

    for col in numeric:
        if named_like_signal(col):
            return col
    for col in numeric:                                    # 退一步：有波动且非时间列
        if any(hint in col.lower() for hint in TIME_HINTS):
            continue
        if float(pd.to_numeric(frame[col], errors="coerce").std() or 0) > 0:
            return col
    return numeric[0]


def read_signal(path: Path | str, column: str | None = None,
                sheet: str | int | None = None) -> np.ndarray:
    """取一列数值作为振动信号（NaN 丢弃）。"""
    frame = read_table(path, sheet=sheet)
    col = pick_signal_column(frame, column)
    values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"列 {col!r} 全为空，取不到信号")
    return values


# ------------------------------------------------------------------ 预览/体检
def _jsonable(value):
    """把 numpy / pandas 的标量转成能进 JSON 的 Python 原生值（NaN → None）。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def preview(path: Path | str, rows: int = 20, sheet: str | int | None = None,
            column: str | None = None) -> dict:
    """表格预览：行列数、每列统计、前 N 行、推荐信号列。"""
    path = Path(path)
    rows = min(max(int(rows), 1), 200)
    frame = read_table(path, sheet=sheet)
    numeric = numeric_columns(frame)
    signal_column = None
    signal_error = None
    try:
        signal_column = pick_signal_column(frame, column)
    except Exception as exc:
        signal_error = str(exc)

    columns = []
    for col in frame.columns:
        series = frame[col]
        conv = pd.to_numeric(series, errors="coerce")
        is_num = str(col) in numeric
        columns.append({
            "name": str(col), "dtype": str(series.dtype), "numeric": is_num,
            "non_null": int(series.notna().sum()), "nulls": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "min": round(float(conv.min()), 6) if is_num and conv.notna().any() else None,
            "max": round(float(conv.max()), 6) if is_num and conv.notna().any() else None,
            "mean": round(float(conv.mean()), 6) if is_num and conv.notna().any() else None,
            "std": round(float(conv.std()), 6) if is_num and conv.notna().any() else None,
            "is_signal": str(col) == signal_column,
        })

    head = [[_jsonable(v) for v in row] for row in frame.head(rows).itertuples(index=False, name=None)]
    return {
        "file": path.name, "path": str(path), "format": path.suffix.lower().lstrip("."),
        "sheet": sheet if sheet is not None else (list_sheets(path) or [None])[0],
        "sheets": list_sheets(path),
        "rows": int(frame.shape[0]), "cols": int(frame.shape[1]),
        "columns": columns, "column_names": [str(c) for c in frame.columns],
        "numeric_columns": numeric,
        "signal_column": signal_column, "signal_error": signal_error,
        "head": head, "head_rows": len(head),
        "size_kb": round(path.stat().st_size / 1024, 1),
    }


def _ttl_cache(max_age: float = 120.0):
    """给「读文件做体检」这类函数加个简单的 TTL 缓存。

    表格数据集的体检要真读一遍 CSV/xlsx（4 万行 × 4 个文件 ≈ 1.3 秒），
    而 /datasets 每次进页面都会调用它——这是页面跳转慢的头号原因。
    """
    import time as _time            # 局部导入，避免动文件头的 import 区
    store: dict = {}

    def deco(fn):
        def wrapper(*args, **kwargs):
            """按 (位置参数, 排序后的关键字参数) 做键查缓存，过期或没有就真跑一遍。"""
            key = (str(args), str(sorted(kwargs.items())))
            hit = store.get(key)
            if hit and _time.time() - hit[0] < max_age:
                return hit[1]
            val = fn(*args, **kwargs)
            store[key] = (_time.time(), val)
            return val
        # 写操作（上传/覆盖数据集文件）之后必须能立刻看到新内容，所以给使用方一个清缓存入口
        wrapper.cache_clear = store.clear
        return wrapper
    return deco


@_ttl_cache(120)
def describe_directory(directory: Path | str, sheet: str | int | None = None,
                       column: str | None = None) -> dict:
    """表格数据集目录体检：每个文件的行数/列数/信号列，以及类别是否重名。"""
    directory = Path(directory)
    files = list_table_files(directory)
    items, errors = [], []
    for path in files:
        item = {"filename": path.name, "label": label_from_filename(path.name),
                "size_kb": round(path.stat().st_size / 1024, 1)}
        try:
            frame = read_table(path, sheet=sheet)
            sig_col = pick_signal_column(frame, column)
            values = pd.to_numeric(frame[sig_col], errors="coerce")
            item.update({
                "rows": int(frame.shape[0]), "cols": int(frame.shape[1]),
                "signal_column": sig_col, "on_disk": True,
                "samples_in_file": int(values.notna().sum()),
                "columns": [str(c) for c in frame.columns],
                "numeric_columns": numeric_columns(frame),
                "min": round(float(values.min()), 6), "max": round(float(values.max()), 6),
            })
        except Exception as exc:
            item.update({"on_disk": True, "error": f"{type(exc).__name__}: {exc}"})
            errors.append(f"{path.name}: {exc}")
        items.append(item)

    labels = [i["label"] for i in items if not i.get("error")]
    duplicates = sorted({lab for lab in labels if labels.count(lab) > 1})
    rows_list = [i["rows"] for i in items if i.get("rows")]
    return {
        "dataset_dir": str(directory),
        "dataset_type": "tabular",
        "file_count": len(items), "classes": len(set(labels)),
        "min_rows": min(rows_list) if rows_list else 0,
        "max_rows": max(rows_list) if rows_list else 0,
        "duplicate_labels": duplicates,
        "files": items, "errors": errors,
        "supported": sorted(TABLE_SUFFIXES),
    }


# ------------------------------------------------------------------ 切窗训练
def load_windows(directory: Path | str, length: int, number: int, stride: int, rate: list[float],
                 normal: bool = True, seed: int = 42, strict: bool = True, legacy_scaler: bool = False,
                 column: str | None = None, sheet: str | int | None = None) -> dict:
    """按「一文件一类别」切窗，返回结构与 `datasets.load_windows` 完全一致。"""
    directory = Path(directory)
    files = list_table_files(directory)
    if not files:
        raise FileNotFoundError(f"{directory} 下没有表格文件（支持 {sorted(TABLE_SUFFIXES)}）")

    samp_train = int(number * (1 - (rate[1] + rate[2])))
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    per_class: list[dict] = []
    labels: list[str] = []

    for class_id, path in enumerate(files):
        signal = read_signal(path, column=column, sheet=sheet)
        # 复用 .mat 数据源同一套切窗（含越界检查），保证两个数据源口径一致
        tr, te, skip_tr, skip_te = ds._slice_windows(signal, number, length, stride, samp_train, strict)
        label = label_from_filename(path.name)
        labels.append(label)
        train_x.extend(tr)
        train_y.extend([class_id] * len(tr))
        test_x.extend(te)
        test_y.extend([class_id] * len(te))
        per_class.append({
            "filename": path.name, "class_id": class_id, "label": label,
            "samples_in_file": int(signal.size),
            "train_windows": len(tr), "test_windows": len(te),
            "skipped_out_of_range": {"train": skip_tr, "test": skip_te},
            "nan_windows": int(sum(1 for w in tr + te if w.size != length)),
        })

    stats = {
        "dataset_dir": str(directory), "dataset_type": "tabular",
        "strict": strict, "legacy_scaler": legacy_scaler, "seed": seed,
        "signal_column": column, "sheet": sheet,
        "num_classes": len(files),
        "nan_windows_total": int(sum(row["nan_windows"] for row in per_class)),
        "skipped_out_of_range_total": int(sum(
            row["skipped_out_of_range"]["train"] + row["skipped_out_of_range"]["test"] for row in per_class)),
        "per_class": per_class,
    }
    return ds.finalize_windows(train_x, train_y, test_x, test_y, labels, rate, normal, seed,
                               legacy_scaler, stats)
