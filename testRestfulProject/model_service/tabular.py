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
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from . import datasets as ds
# 支持的表格后缀（决定"这个目录算不算表格数据集"）；EXCEL_SUFFIXES 是其中要交给 Excel 引擎的那部分。
TABLE_SUFFIXES = {".csv", ".txt", ".xlsx", ".xlsm", ".xls"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
# 选信号列用的"名字线索"。⚠️ 判定是**子串包含**（hint in name.lower()）而不是精确匹配，
# 所以 "de" 这类短线索会误伤 model_id、code 之类的列名（"index" 同样含 de，但会先被 TIME_HINTS 拦下）。
# 这套线索也解释了 DEMO 数据（列名：时间 / 振动幅值 / 温度）里为什么"振动幅值"胜出。
SIGNAL_HINTS = ("振幅", "振动", "幅值", "加速度", "signal", "value", "amplitude", "vibration",
                "de_time", "de", "acc")
# 时间/序号列优先排除：它们天然单调递增、std 也很大，是最容易被误选成"振动信号"的干扰项。
TIME_HINTS = ("时间", "时刻", "序号", "采样点序号", "time", "timestamp", "date", "index", "no.")
# "重猜分隔符"时**只认这几种**。为什么要有白名单：`csv.Sniffer` 会把表头里的普通字母也当候选分隔符，
# 单列表头 `signal` 会被按字母 `s` 切开（列名变成 `Unnamed: 0` + `ignal`），而列数确实从 1 变成 2，
# 光靠"猜出来列更多才采纳"根本挡不住 —— 实测 `signal` / `value` 都会中招。
# 分隔符只可能是这几个符号，把白名单卡在这里最省事，也不影响正常的分号/制表符数据。
_PLAUSIBLE_SEPS = (",", ";", "\t", "|")
def _sniff_sep(text: str) -> str | None:
    """猜 CSV 分隔符；猜不出或猜出来的不是常见分隔符就返回 None（调用方据此保持原样）。"""
    try:
        sep = csv.Sniffer().sniff(text).delimiter
    except Exception:                       # Sniffer 对单列/不规则文本会抛"Could not determine delimiter"
        return None
    return sep if sep in _PLAUSIBLE_SEPS else None
def is_table(path: Path | str) -> bool:
    """是不是一张可读的表格（只看扩展名，不打开文件）。"""
    # 只看后缀不打开文件：便宜到可以在遍历目录时随便调；代价是坏文件/空文件也会被认成表格，
    # 真正的读取错误留到 read_table / describe_directory 里报出来。
    return Path(path).suffix.lower() in TABLE_SUFFIXES
def has_tables(directory: Path | str) -> bool:
    """目录里是否存在表格文件——训练侧据此自动判定数据源类型（matlab / tabular）。"""
    # ⚠️ 只回答"目录里有没有表格后缀的文件"，不回答"目录存不存在"——目录存在性是上层
    #    training.resolve_source 先校验的（那里会报"数据集目录不存在"）。
    # ⚠️ 上层判定数据源时是**先问表格、再找 .mat**：目录里同时放 .mat 和 csv 会走表格路径（见 resolve_source）。
    directory = Path(directory)
    return directory.is_dir() and any(is_table(p) for p in directory.iterdir() if p.is_file())
def list_table_files(directory: Path | str) -> list[Path]:
    """按文件名排序返回表格文件（排序即类别号顺序，稳定可复现）。"""
    # ⚠️ 这个 list 顺序被 load_windows 直接当 class_id 用（enumerate），所以排序方式是"类别号口径"的一部分，
    #    改排序 = 改历史实验的类别编号。用 name.lower() 是为了不受大小写影响地稳定排序；
    #    但仅大小写不同的两个文件（a.csv / A.xlsx）在该键下并列，次序会落回 iterdir 的返回顺序，
    #    这种情况请自己把文件名改出区别。
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and is_table(p)],
                  key=lambda p: p.name.lower())
def label_from_filename(filename: str) -> str:
    """类别名 = 文件名去扩展名（不做其它改写，保证可追溯）。"""
    # 刻意不做任何清洗（不去空格、不替换下划线）：标签要能一眼对回磁盘上的文件。
    # ⚠️ 代价是 a.csv 与 a.xlsx 会得到同名标签 → 两个文件被当成两个"同名类别"，
    #    describe_directory 的 duplicate_labels 就是用来把这情况暴露出来的。
    return Path(filename).stem
# ------------------------------------------------------------------ 读表
def read_table(path: Path | str, sheet: str | int | None = None) -> pd.DataFrame:
    """读成 DataFrame。Excel 按后缀挑引擎，CSV 依次试编码并在必要时重猜分隔符。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in EXCEL_SUFFIXES:
        # .xls 是老格式，只有 xlrd 能读；.xlsx/.xlsm 交给 openpyxl
        # （.xls 是二进制老格式，.xlsx/.xlsm 是 zip+xml 新格式，两套引擎完全不通用，选错直接报错）
        # sheet 默认取第 0 个：调用方（预览页）会先把 list_sheets 的结果给用户选
        engine = "xlrd" if suffix == ".xls" else "openpyxl"
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, engine=engine)
    last_error = None
    # 编码顺序是照中文 Excel 导出的实际情况排的：utf-8-sig（带 BOM）最保险，GBK 兜底
    # ⚠️ 顺序不能反着来：GBK 几乎能"吃下"任意字节而不报错，排在前面会把 UTF-8 中文静默读成乱码。
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            if frame.shape[1] == 1:                       # 可能是分号/制表符分隔
                # ⚠️ 只在"读出来只有一列"时才重猜分隔符，且必须**两个条件同时成立**才采纳：
                #    ① 猜出来的列更多；② 猜出来的分隔符在 _PLAUSIBLE_SEPS 白名单里。
                #    条件②是踩过的坑：不加它的话 Sniffer 会把表头里的普通字母当分隔符，
                #    单列表头 `signal` 被按 `s` 切开、列数从 1 变 2，条件①反而"通过"了。
                # ⚠️ 重猜失败**绝不能连累已经读成功的结果**：`正常\\n1.0\\n2.0\\n3.0` 这种中文单列表头
                #    会让 Sniffer 抛 "Could not determine delimiter"，以前那个异常直接 break 掉整个循环，
                #    把一份完全正常的单列数据变成"读取失败"。所以这里自己 try 住、失败就原样返回。
                try:
                    sample = path.read_text(encoding=encoding, errors="replace")[:4096]
                except Exception:
                    sample = ""
                sep = _sniff_sep(sample)
                if sep:
                    try:
                        alt = pd.read_csv(path, encoding=encoding, sep=sep)
                    except Exception:
                        alt = None
                    if alt is not None and alt.shape[1] > frame.shape[1]:
                        frame = alt
            return frame
        except UnicodeDecodeError as exc:                  # 换编码重试
            last_error = exc
            continue
        except Exception as exc:                           # 其它错误（空文件、格式坏）换编码也没用
            # 只有编码问题才值得换下一种编码；其它异常直接 break，避免同一份坏文件被读三遍
            last_error = exc
            break
    raise ValueError(f"读取失败：{path.name}（{last_error}）")
def list_sheets(path: Path | str) -> list[str]:
    """Excel 的 sheet 名列表（非 Excel 或读失败一律返回空列表，调用方无需处理异常）。"""
    # 刻意吞掉所有异常：这个方法只服务于"给个 sheet 下拉框"，文件坏了不该让整个预览接口 500，
    # 真正的读取错误会在 read_table 里抛出并显示给用户。
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
    # 分两条路判断：pandas 已经认成数字 dtype 的直接收；object 列则尝试转换，
    # 允许少量脏值（≥80% 能转成数字就算），因为实际 CSV 里常混着表头注释、单位行、"N/A"。
    # 统一 str(col) 输出：Excel 的表头可能是 0/1 这种 int，而调用方传进来的列名是字符串，
    # 不统一类型就会出现"明明有这列却说列不存在"。
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
    # 显式指定优先：用户传了 column 就以它为准，但要先确认"这列存在且是数值列"，
    # 否则后面 to_numeric 会得到整列 NaN，报错现场离真正的原因很远（这里提前拦下并把候选列回显出来）。
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
        # 第一优先级：列名像信号。以 DEMO 轴承表格数据（列：时间/振动幅值/温度）为例，
        # "振动幅值"命中 SIGNAL_HINTS 且不命中 TIME_HINTS → 被选中；
        # "时间"命中 TIME_HINTS 被直接排除；"温度"两边都不命中，只能靠下面的兜底规则去处理。
        if named_like_signal(col):
            return col
    for col in numeric:                                    # 退一步：有波动且非时间列
        # ⚠️ 这条兜底会误伤："温度"这类不含任何线索的物理量（std > 0、名字里也没有 time/index）
        #    同样满足条件——一旦表里没有"振动幅值"这类列，温度就可能被当成振动信号送进模型。
        #    所以真实数据要么规范列名，要么在调用/训练时显式传 column=。
        if any(hint in col.lower() for hint in TIME_HINTS):
            continue
        if float(pd.to_numeric(frame[col], errors="coerce").std() or 0) > 0:
            return col
    # 最后兜底：一个都没挑出来就取第一个数值列，至少保证流程能往下走（而不是抛异常卡死在预检）
    return numeric[0]
def read_signal(path: Path | str, column: str | None = None,
                sheet: str | int | None = None) -> np.ndarray:
    """取一列数值作为振动信号（NaN 丢弃）。"""
    frame = read_table(path, sheet=sheet)
    col = pick_signal_column(frame, column)
    # ⚠️ dropna() 会**静默丢点**：空单元格、非数值文本都会被 to_numeric 变成 NaN 后丢掉，
    #    于是"信号里第 k 个点"不再等于"表里第 k 行"，窗口 ↔ 行号 的对应关系就此断裂——
    #    后续想把某个异常窗口回溯到具体时间/行号时，不能直接拿行号去索引原表。
    #    （体检里的 samples_in_file 也是按非空计数，和 rows 不是一回事。）
    # ⚠️ inf 不会被 dropna 拦住：它会一路带进标准化，算出 NaN 的均值/方差，最终整批样本变 NaN。
    values = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"列 {col!r} 全为空，取不到信号")
    return values
# ------------------------------------------------------------------ 预览/体检
def _jsonable(value):
    """把 numpy / pandas 的标量转成能进 JSON 的 Python 原生值（NaN → None）。"""
    # 存在理由：DataFrame 里的值多是 np.int64 / np.float64 / pd.Timestamp，Flask 的 jsonify
    # 遇到它们会直接抛 TypeError；而 NaN/NaT 也不是合法 JSON 字面量（前端 JSON.parse 会报错），
    # 所以统一在这里转成 None。
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
    # 前端可传 rows（预览行数），这里夹到 1..200：既挡住 rows=1000000 把页面拖死，
    # 也把 rows=0/负数 这类输入收敛成"至少 1 行"，不让 head 变成空表。
    rows = min(max(int(rows), 1), 200)
    frame = read_table(path, sheet=sheet)
    numeric = numeric_columns(frame)
    signal_column = None
    signal_error = None
    # 选列失败不该让整个预览接口挂掉：把异常信息放进 signal_error 一并返回，
    # 前端照样能显示列统计与原始行，只是"推荐信号列"为空并附上原因。
    try:
        signal_column = pick_signal_column(frame, column)
    except Exception as exc:
        signal_error = str(exc)
    # 逐列算一份统计供表格预览页展示：空值/非空/唯一值，以及数值列的 min/max/mean/std。
    # is_num 复用 numeric_columns（与选信号列同一口径），conv 只用来算统计量。
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
    # 前 N 行原样交给前端：itertuples(name=None) 拿纯值元组（不会被 pandas 塞进索引列），
    # 每个格子再经 _jsonable 转成原生类型，否则 np.int64/Timestamp 会让 jsonify 抛错。
    head = [[_jsonable(v) for v in row] for row in frame.head(rows).itertuples(index=False, name=None)]
    return {
        "file": path.name, "path": str(path), "format": path.suffix.lower().lstrip("."),
        # ⚠️ list_sheets 在这里被调了两次（回显当前 sheet + sheet 列表），等于多解析一次 Excel 头；
        #    预览是低频操作，这点成本换"前端一次拿到全部下拉项"是划算的。
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
    # 缓存就挂在装饰器闭包里（一个被装饰函数一个 store），进程级、无淘汰、非线程安全。
    # ⚠️ 多线程同时未命中会各体检一遍，结果一致只是白做一次；条目数等于参数组合数，量级很小。
    store: dict = {}
    def deco(fn):
        def wrapper(*args, **kwargs):
            """按 (位置参数, 排序后的关键字参数) 做键查缓存，过期或没有就真跑一遍。"""
            # ⚠️ 键是参数的字符串形式：同一个目录传 Path 还是 str、参数用位置还是关键字传，
            #    都会算成不同的键（各自体检一遍，正确性没问题，只是白读文件）。
            # ⚠️ 键里不含文件 mtime，所以 TTL 内覆盖/上传了文件仍会读到旧结果 →
            #    写操作之后必须显式调 cache_clear()（见 api.py 的上传接口）。
            key = (str(args), str(sorted(kwargs.items())))
            hit = store.get(key)
            if hit and _time.time() - hit[0] < max_age:
                return hit[1]
            val = fn(*args, **kwargs)
            store[key] = (_time.time(), val)
            return val
        # 写操作（上传/覆盖数据集文件）之后必须能立刻看到新内容，所以给使用方一个清缓存入口
        # （api.py 上传成功后第一时间调 tabular.describe_directory.cache_clear()，否则列表页
        #   还会拿旧内容显示 120 秒）
        wrapper.cache_clear = store.clear
        return wrapper
    return deco
@_ttl_cache(120)
def describe_directory(directory: Path | str, sheet: str | int | None = None,
                       column: str | None = None) -> dict:
    """表格数据集目录体检：每个文件的行数/列数/信号列，以及类别是否重名。"""
    # ⚠️ 这个函数会**真读一遍**每个文件（4 万行 × 4 个文件 ≈ 1.3 秒），而 /datasets 每次进页面都调它，
    #    所以上面必须挂 @_ttl_cache(120)：这是页面跳转慢的头号原因，别把装饰器删了。
    directory = Path(directory)
    files = list_table_files(directory)
    items, errors = [], []
    for path in files:
        # 一个文件一个体检项：label = 文件名去扩展名（即类别名），先填基本信息，读成功后再补行/列统计
        item = {"filename": path.name, "label": label_from_filename(path.name),
                "size_kb": round(path.stat().st_size / 1024, 1)}
        try:
            # 这里必须真的读表才能知道列名与信号列（行数/列数/是否数值列都无法只读元数据得到）
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
            # 单个文件坏掉不影响整次体检：只把它标成 error（on_disk 仍为 True，文件确实在），
            # 并记进 errors 列表，页面提示"这个文件读不了"而不是整个接口 500。
            item.update({"on_disk": True, "error": f"{type(exc).__name__}: {exc}"})
            errors.append(f"{path.name}: {exc}")
        items.append(item)
    # 重名检测只看读成功的文件（读失败的文件连标签都不可信）；
    # labels.count 是 O(n²)，但文件数是"一个类别一个文件"的量级，完全够用。
    labels = [i["label"] for i in items if not i.get("error")]
    duplicates = sorted({lab for lab in labels if labels.count(lab) > 1})
    # min/max_rows 只统计读成功且有行数的项；一个都没有时给 0，避免对空列表调 min() 抛异常
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
    # files 是排好序的列表，下面的 enumerate 直接把序号当 class_id →
    # 类别号 = 文件名排序位次（这个顺序决定了 labels 与报告里的编号，别随意改排序规则）。
    files = list_table_files(directory)
    if not files:
        raise FileNotFoundError(f"{directory} 下没有表格文件（支持 {sorted(TABLE_SUFFIXES)}）")
    # 与 .mat 数据源一样：训练窗口数由 number 与"验证+测试"占比反推，rate[0] 不参与运算
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
        # （训练窗从 j*stride 起、测试窗从 samp_train*stride+length 起，越界窗口跳过、不补 NaN）
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
    # stats 字段名刻意与 .mat 数据源一一对齐（num_classes / nan_windows_total /
    # skipped_out_of_range_total / per_class …），上层训练接口、库表备注、前端展示因此都不用
    # 区分数据源类型；多出来的 signal_column / sheet 是表格特有的溯源信息。
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
    # 收尾（标准化 / 验证+测试分层划分 / 打乱训练集）与 .mat 通路完全一致，直接复用同一份实现，
    # 避免两条数据通路各写一套导致口径漂移
    return ds.finalize_windows(train_x, train_y, test_x, test_y, labels, rate, normal, seed,
                               legacy_scaler, stats)
