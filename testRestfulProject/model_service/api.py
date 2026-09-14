# -*- coding: utf-8 -*-
"""flask_restful 接口资源（流程图里的「Web访问」盒子）。

路由一览：

    GET  /api                      接口索引
    GET  /health                   服务 / 数据库 / 模型产物体检
    GET  /models                   模型清单（落盘产物 + 库表登记）
    GET  /datasets                 数据集体检（含每个文件的点数，暴露 IR014 偏短这类问题）
    POST /train                    训练一个模型 → 落盘 + 写 Trainings
    GET  /trainings                最近训练记录（读库）
    POST /predict                  推理 → 写 InferenceTasks + InferenceResults
    GET  /inference-tasks          最近推理任务（读库）
    GET  /inference-tasks/<id>     单个任务及其结果明细（读库）

约定：任何失败都返回 {"error": ...} + 合适的状态码，并把细节写进 ModelInvocations（能写库时）。
"""

from __future__ import annotations

import json
import platform
import re
import sys
import traceback
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from flask import Response, redirect, request, send_from_directory
from flask_restful import Resource

from . import datasets as ds
from . import tabular
from .config import config
from .db import DBError, database
from .figures import FIG_DIR, clear_figures, list_figures
from .inference import InvalidInput, predict
from .registry import delete_version, list_artifacts, load_artifact
from .training import MODEL_META, ALIASES, normalize_model, train

CONSOLE_HTML = Path(__file__).with_name("console.html")

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")          # 训练日志里 Keras 进度条的转义序列
_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "h5py", "flask",
             "flask-restful", "pymysql", "pyodbc", "tensorflow", "tf-nightly", "keras",
             "keras-nightly", "torch", "python-docx", "pywin32")

MAX_EPOCHS = 200          # 防止一条 HTTP 请求把服务占住几小时
MAX_LIMIT = 500           # 单次 /predict 最多多少窗口

# ---------------- 「这是不是一个模型」探测 ----------------
# 上传时不再要求用户填输入长度/类别数/类别标签：先按扩展名分类，再**打开文件看内容**，
# 判断它到底是不是权重文件，并尽量把 input_len / num_classes 猜出来。
WEIGHT_SUFFIXES = {
    ".h5": "tensorflow-keras", ".keras": "tensorflow-keras",
    ".pt": "pytorch", ".pth": "pytorch",
    ".pkl": "adtk", ".pickle": "adtk",
}
# 上传文件夹时，这些扩展名之外的文件一律忽略（允许带上 scaler/meta/说明文件等附属文件）
KEEP_SUFFIXES = {".json", ".npz", ".npy", ".txt", ".yaml", ".yml", ".onnx", ".csv",
                 ".h5", ".keras", ".pt", ".pth", ".pkl", ".pickle"}
# PyTorch 的输出层一般叫这些名字，用来从 state_dict 里认出"最后一层"从而读出类别数
_HEAD_LAYER_RE = re.compile(r"(fc|classifier|linear|head|dense|out|output)\d*\.weight$")


def _keras_shapes(config: dict) -> tuple[int | None, int | None]:
    """从 Keras 的 model_config（dict）里挖出 input_len 与类别数（最后一个 Dense 的 units）。"""
    conf = config.get("config", config) if isinstance(config, dict) else None
    if not isinstance(conf, dict):
        return None, None
    layers = conf.get("layers") if isinstance(conf.get("layers"), list) else []

    def first_dim(shape) -> int | None:
        """取形状里第一个有意义的维度。

        跳过 batch 维（shape[0]，训练时是 None）与 None/0 这类占位维度，
        所以 [None, 784, 1] 会返回 784 而不是 None。
        """
        if not isinstance(shape, list):
            return None
        dims = [d for d in shape[1:] if isinstance(d, int) and d > 0]
        return int(dims[0]) if dims else None

    input_len = None
    for layer in layers:                                   # 输入层：Keras2 用 batch_input_shape，Keras3 用 batch_shape
        lc = layer.get("config") if isinstance(layer, dict) else None
        if isinstance(lc, dict):
            input_len = first_dim(lc.get("batch_input_shape")) or first_dim(lc.get("batch_shape"))
            if input_len:
                break
    if input_len is None:                                  # 顶层也可能直接挂形状
        input_len = first_dim(conf.get("batch_input_shape")) or first_dim(conf.get("batch_shape"))
    units = None
    for layer in reversed(layers):                         # 最后一个 Dense 的 units 就是类别数
        lc = layer.get("config") if isinstance(layer, dict) else None
        if isinstance(lc, dict) and isinstance(lc.get("units"), int) and lc["units"] > 0:
            units = int(lc["units"])
            break
    return input_len, units


def probe_weight(filename: str, blob: bytes) -> dict:
    """判断一个上传文件是不是模型权重，并尽量读出 input_len / num_classes。

    返回 {ok, framework, reason, input_len, num_classes}。判定规则（看内容，不只看后缀）：

      .h5      HDF5，且含 `model_weights` 组或 `model_config` 属性（Keras 存档）
      .keras   zip，且含 config.json / metadata.json（Keras 3 存档）
      .pt/.pth zip 且含 torch 的 data.pkl；能反序列化就顺手把类别数读出来
      .pkl     pickle（0x80 开头）——**不反序列化**，免得「上传即执行代码」
    """
    import io
    import zipfile

    suffix = Path(filename).suffix.lower()
    framework = WEIGHT_SUFFIXES.get(suffix)
    result = {"ok": False, "framework": framework, "reason": "", "input_len": None, "num_classes": None}
    if framework is None:
        result["reason"] = f"扩展名 {suffix or '(无)'} 不是模型权重（支持 .h5/.keras/.pt/.pth/.pkl/.pickle）"
        return result
    if not blob:
        result["reason"] = "文件是空的（0 字节）"
        return result

    try:
        if suffix in (".h5", ".keras"):
            if blob[:2] == b"PK":                          # Keras 3 的 .keras 是 zip
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    names = zf.namelist()
                    if "config.json" not in names and "metadata.json" not in names:
                        result["reason"] = f"压缩包里没有 config.json/metadata.json，不像 Keras 模型（含 {names[:5]}）"
                        return result
                    config = json.loads(zf.read("config.json").decode("utf-8"))
                input_len, units = _keras_shapes(config)
                result.update(ok=True, input_len=input_len, num_classes=units,
                              reason=f"Keras 3 存档（zip，input_len={input_len}，类别数={units}）")
                return result
            import h5py
            with h5py.File(io.BytesIO(blob), "r") as handle:
                keys = list(handle.keys())
                raw_config = handle.attrs.get("model_config")
                if "model_weights" not in keys and raw_config is None:
                    result["reason"] = (f"HDF5 里既没有 model_weights 组也没有 model_config 属性"
                                        f"（顶层是 {keys[:5]}），不像 Keras 权重文件")
                    return result
                config = json.loads(raw_config) if raw_config is not None else None
            input_len, units = _keras_shapes(config) if config else (None, None)
            result.update(ok=True, input_len=input_len, num_classes=units,
                          reason=f"Keras HDF5（顶层 {keys[:5]}，input_len={input_len}，类别数={units}）")
            return result

        if suffix in (".pt", ".pth"):
            if blob[:2] != b"PK":
                result["reason"] = (f"不是 PyTorch 检查点（torch>=1.6 的 .pt 是 zip，应以 PK 开头；"
                                    f"实际开头 {blob[:4]!r}）")
                return result
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                names = zf.namelist()
            if not any(name.endswith("data.pkl") for name in names):
                result["reason"] = f"zip 里没有 torch 的 data.pkl（含 {names[:5]}），不像 PyTorch 模型"
                return result
            try:
                import torch
            except Exception as exc:                       # 本机没装 torch 也不该误判成"不是模型"
                result.update(ok=True, reason=f"PyTorch 检查点（本机无 torch，跳过参数解析：{exc}）")
                return result
            try:
                obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            except Exception as exc:                       # 存成完整模型对象时解析不了，但仍按权重接受
                result.update(ok=True, reason=f"PyTorch 检查点（未解析参数：{type(exc).__name__}）")
                return result
            state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
            shapes = [(k, tuple(v.shape)) for k, v in state.items()] if hasattr(state, "items") else []
            if not shapes:
                result["reason"] = "torch 文件里没有张量参数，不像模型权重"
                return result
            head = next((s for k, s in reversed(shapes) if len(s) == 2 and _HEAD_LAYER_RE.search(k)), None)
            head = head or next((s for _, s in reversed(shapes) if len(s) == 2), None)
            result.update(ok=True, num_classes=int(head[0]) if head else None,
                          reason=f"PyTorch state_dict（{len(shapes)} 个张量，类别数={int(head[0]) if head else None}）")
            return result

        if blob[:1] != b"\x80":                            # pickle 协议 2+ 都以 0x80 开头
            result["reason"] = f"不是 pickle 文件（.pkl 应以 0x80 开头，实际 {blob[:2]!r}）"
            return result
        result.update(ok=True, reason="pickle 序列化对象（adtk 检测器/传统模型；为安全起见不做反序列化校验）")
        return result
    except Exception as exc:                               # 坏文件、半截文件都会走到这里
        result["reason"] = f"读取失败，不像有效模型文件：{type(exc).__name__}: {exc}"
        return result


def _body() -> dict:
    """请求体 JSON；不是对象（或压根不是 JSON）时返回空 dict，避免调用方到处判 None。

    用 `silent=True` 是刻意的：前端有些请求是 form-data 或不带 body，
    这时不该直接 400，而应让接口用参数默认值继续跑。
    """
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _int(value, default=None, name="参数"):
    """取整数参数：None 用默认值，转不动就抛 InvalidInput（接口据此回 400）。

    ⚠️ 调用方**不要**写成 `_int(x, 0, "y") or 0`：0 是 falsy，会把用户明确传的 0
    悄悄换成别的值 —— `/predict` 的 `limit=0` 曾因此绕过范围校验。
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise InvalidInput(f"{name} 必须是整数，收到 {value!r}")


def _resolve_workspace_path(raw: str) -> Path:
    """把用户给的路径解析成**工作区内**的文件（`/datasets/table`、`/datasets/signal` 用它）。

    两级回退：相对路径先按工作区（D:\\22project）试，再按项目目录（testRestfulProject）试，
    所以 `data/datasets/x.csv` 与 `testRestfulProject/data/datasets/x.csv` 两种写法都能用。

    安全：候选路径必须满足 `relative_to(workspace)`，用的是 Path 语义而**不是字符串
    startswith** —— 否则 `D:\\22project_evil\\x` 这类同前缀目录会绕过检查（历史上真踩过）。
    最终找不到就抛 InvalidInput(400)，而不是把原始路径回显出去让人猜。
    """
    candidate = Path(raw)
    # 绝对路径只有一次机会；相对路径两个基准各试一次（工作区优先）
    tries = [candidate] if candidate.is_absolute() else \
        [config.workspace_dir / candidate, config.project_dir / candidate]
    for path in tries:
        resolved = path.resolve()
        try:
            resolved.relative_to(config.workspace_dir.resolve())    # 越界 → 换下一个候选
        except ValueError:
            continue
        if resolved.is_file():            # 目录不算命中，必须是指到文件
            return resolved
    raise InvalidInput(f"文件不存在或不在工作区内：{raw}")


class ApiIndex(Resource):
    """GET /api —— 接口索引，给人和「系统管理→接口索引」页看的自描述清单。"""

    def get(self):
        return {
            "service": "model_service",
            "flow": "Web访问 → 算法模型 → 训练 → 数据集 → 模型产物 → 推理 → (边缘设备)",
            "endpoints": {
                "GET /ui": "单页控制台（首页/模型管理/数据集管理/数据展示/系统管理）",
                "GET /api": "本索引",
                "GET /health": "服务/数据库/产物体检",
                "GET /models": "模型清单（产物 + 库表）",
                "GET /models/<model>": "产物 meta.json；DELETE ?version=vN 删除该版本",
                "GET /datasets": "数据集体检（内置 .mat + 上传的表格数据集）",
                "GET /datasets/db": "Datasets 表登记记录；POST 登记新数据集",
                "POST /datasets/upload": "上传表格文件到 data/datasets/<名称>/（multipart，字段 name + file）",
                "GET /datasets/table": "表格预览（列统计/前 N 行/推荐信号列，?path=）",
                "GET /datasets/signal": "取一段原始信号（.mat 或表格，画波形用）",
                "POST /train": "训练并落盘（model, epochs, dataset, ...）",
                "GET /trainings": "最近训练记录（库）",
                "POST /predict": "推理（model, samples 或 path）",
                "GET /inference-tasks": "最近推理任务（库）",
                "GET /inference-tasks/<id>": "任务与结果明细（库）",
                "GET /figures": "已生成的图（训练曲线/混淆矩阵/预测分布…）",
                "GET /figures/<路径>": "直接看某张 PNG",
                "GET /system": "运行信息（Python/依赖版本/路径/行数/占用）",
                "GET /system/logs": "训练日志列表；GET /system/logs/<name>?tail=N 看尾部",
                "POST /system/maintenance": "维护操作（目前支持清空图库 target=figures）",
                "GET/POST/PUT/DELETE /todos": "原有的示例接口，保留不动",
            },
            "models": sorted(set(ALIASES.values())),
        }


class Health(Resource):
    """GET /health —— 服务 / 数据库 / 模型产物的体检，前端顶栏每 15 秒轮询它。

    刻意**不抛异常**：数据库连不上也算"服务还活着"，只在 database.ok 里标失败，
    否则顶栏会显示成"服务不可用"，让人误以为整个进程挂了。
    """

    def get(self):
        db_state = database.ping()
        artifacts = list_artifacts()
        return {
            "service": "ok",
            "config": config.describe(),
            "database": db_state,
            "artifacts": {
                "count": len(artifacts),
                "by_model": {name: [a.version for a in artifacts if a.name == name]
                             for name in sorted({a.name for a in artifacts})},
            },
            "figures": {"count": len(list_figures(limit=1000)), "dir": str(FIG_DIR)},
            "warnings": [
                "本服务不做训练/推理排队，/train 是同步阻塞的（开发服务器已开 threaded）。",
                "数据管线已知问题：原脚本会把越界切片补成整行 NaN；服务侧改为拒绝并回报。",
            ],
        }


class ModelList(Resource):
    """GET /models —— 磁盘产物（artifacts）+ 库表登记（db_models）两份清单。

    两者故意不合并：产物可能还没登记（训练落盘后写库失败），登记也可能没有产物
    （只占位未训练），前端要能看出这种不一致。
    """

    def get(self):
        artifacts = list_artifacts()
        try:
            db_models = database.models_in_db()
        except DBError as exc:
            db_models = [{"error": str(exc)}]
        return {
            "artifacts": [a.to_dict() for a in artifacts],
            "db_models": db_models,
            "known_models": {name: MODEL_META[name] for name in MODEL_META},
        }


class DatasetList(Resource):
    """数据集体检：内置 .mat 数据集 + data/datasets 下上传的表格数据集。"""

    def get(self):
        """把两类数据源都体检一遍。

        单个数据集出错只写进它自己的 error 字段，不让整页 500 —— 某个表格文件
        损坏不应该连带看不到其它数据集。
        """
        out = {}
        for name, path in config.dataset_dirs.items():
            try:
                info = ds.describe_dataset(path)
                info["dataset_type"] = "matlab"
                info["key"] = name
                out[name] = info
            except Exception as exc:
                out[name] = {"error": f"{type(exc).__name__}: {exc}", "dataset_dir": str(path),
                             "dataset_type": "matlab", "key": name}
        upload_root = sorted([p for p in config.upload_dir.iterdir() if p.is_dir()], key=lambda p: p.name) \
            if config.upload_dir.is_dir() else []          # 目录被删掉时不该 500，当成"没有上传数据集"
        for directory in upload_root:
            key = f"表格:{directory.name}"
            try:
                info = tabular.describe_directory(directory)
            except Exception as exc:
                info = {"error": f"{type(exc).__name__}: {exc}", "files": []}
            info.update({"key": key, "dataset_type": "tabular", "dataset_dir": str(directory),
                         "file_count": info.get("file_count", 0), "classes": info.get("classes", 0)})
            out[key] = info
        return out, 200


class DatasetUpload(Resource):
    """上传表格文件到 data/datasets/<数据集名>/（一个文件 = 一个类别）。"""

    MAX_MB = 80

    def post(self):
        """保存上传的表格文件，一个文件 = 一个类别（文件名即标签）。"""
        name = (request.form.get("name") or "").strip()
        files = request.files.getlist("file") or request.files.getlist("files")
        if not files:
            return {"error": "没有收到文件（表单字段名用 file，可重复传多个）"}, 400
        if not name:
            name = Path(files[0].filename or "dataset").stem
        safe_name = re.sub(r"[^\w\u4e00-\u9fa5.\-]+", "_", name).strip("._") or "dataset"
        target = config.upload_dir / safe_name
        target.mkdir(parents=True, exist_ok=True)
        saved, skipped, overwritten = [], [], []
        for item in files:
            filename = Path(item.filename or "").name
            if not filename or Path(filename).suffix.lower() not in tabular.TABLE_SUFFIXES:
                skipped.append({"filename": filename,
                                "reason": f"不支持的扩展名，支持 {sorted(tabular.TABLE_SUFFIXES)}"})
                continue
            blob = item.read()
            if not blob:                                   # 0 字节文件会变成"空类别"，直接拒收
                skipped.append({"filename": filename, "reason": "空文件（0 字节）"})
                continue
            if len(blob) > self.MAX_MB * 1024 * 1024:
                skipped.append({"filename": filename, "reason": f"超过 {self.MAX_MB}MB"})
                continue
            destination = target / filename
            replaced = destination.is_file()               # 同名会被静默覆盖，这里至少回报出来
            destination.write_bytes(blob)
            if replaced:
                overwritten.append(filename)
            saved.append({"filename": filename, "size_kb": round(len(blob) / 1024, 1),
                          "path": str(destination), "label": tabular.label_from_filename(filename),
                          "replaced": replaced})
        # 体检结果有 120 秒 TTL 缓存：不清掉的话，响应里和列表页里都还是上传前的旧内容
        tabular.describe_directory.cache_clear()
        try:
            listing = tabular.describe_directory(target)
        except Exception as exc:
            listing = {"error": str(exc)}
        return {"dataset": safe_name, "directory": str(target), "saved": saved,
                "skipped": skipped, "overwritten": overwritten,
                "hint": ("同名文件已被覆盖：本平台约定「一个文件 = 一个类别、文件名即标签」，"
                         "覆盖会直接换掉那个类别的全部数据" if overwritten else None),
                "listing": listing}, 200 if saved else 400


class TablePreview(Resource):
    """表格预览：列统计 + 前 N 行 + 推荐信号列（数据展示/数据集管理都用它）。"""

    def get(self):
        raw = request.args.get("path")
        if not raw:
            return {"error": "缺少 path 参数"}, 400
        try:
            path = _resolve_workspace_path(raw)
        except InvalidInput as exc:
            return {"error": str(exc)}, 400
        if not tabular.is_table(path):
            return {"error": f"不是可预览的表格文件（支持 {sorted(tabular.TABLE_SUFFIXES)}）：{raw}"}, 400
        try:
            data = tabular.preview(path, rows=_int(request.args.get("rows"), 20, "rows") or 20,
                                   sheet=request.args.get("sheet"), column=request.args.get("column"))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}, 500
        return data, 200


class Train(Resource):
    """POST /train —— 训练入口。

    这一层只做**参数校验与归一化**（模型名、epochs、rate、dataset_dir 的路径安全），
    真正的训练在 training.train() 里同步跑完，所以本接口耗时长、不能并发压。
    """

    def post(self):
        body = _body()
        try:
            name = normalize_model(body.get("model"))
            epochs = _int(body.get("epochs"), None, "epochs")
            if epochs is not None and not (1 <= epochs <= MAX_EPOCHS):
                raise InvalidInput(f"epochs 必须在 1..{MAX_EPOCHS} 之间")
            rate = body.get("rate")
            if rate is not None:                       # rate 不校验会一路炸到除零/空训练集（500）
                if not isinstance(rate, (list, tuple)) or len(rate) != 3:
                    raise InvalidInput("rate 必须是长度为 3 的数组，例如 [0.7,0.15,0.15]")
                try:
                    values = [float(x) for x in rate]
                except (TypeError, ValueError):
                    raise InvalidInput(f"rate 必须是数字数组，收到 {rate!r}")
                # 三项非负且和为 1：否则切出来的数据集大小算不对
                if any(x < 0 for x in values) or abs(sum(values) - 1.0) > 1e-6:
                    raise InvalidInput(f"rate 三项必须非负且和为 1，收到 {rate!r}")
                if values[1] + values[2] <= 0:
                    raise InvalidInput("rate[1]+rate[2] 必须大于 0（验证/测试集不能为空）")
                body["rate"] = values
            options = {k: v for k, v in body.items() if k not in ("model",)}
            for key in ("dataset_type", "dataset_dir", "signal_column", "sheet", "dataset"):
                if body.get(key) is not None:
                    options[key] = body[key]
            if body.get("dataset_dir"):
                candidate = Path(body["dataset_dir"])
                if candidate.is_absolute():
                    resolved = candidate.resolve()
                else:
                    # 相对路径口径与 `_resolve_workspace_path` 一致：先按工作区试，再按项目目录试。
                    # 这一步必须两边都试 —— 响应脱敏后前端拿到的是 "testRestfulProject\\1DCNN\\0HP"
                    # 这种"相对工作区"的路径，只按项目目录拼会得到 testRestfulProject\testRestfulProject\...
                    first = config.workspace_dir / candidate
                    resolved = (first if first.exists() else config.project_dir / candidate).resolve()
                # 用 relative_to 判断"在工作区内"：字符串 startswith 会被 D:\22project_evil
                # 这类同前缀目录绕过（inference._guard_path 也是这么做的，口径统一）
                try:
                    resolved.relative_to(config.workspace_dir.resolve())
                except ValueError:
                    raise InvalidInput("dataset_dir 必须在工作区内")
                options["dataset_dir"] = str(resolved)
            if epochs is not None:
                options["epochs"] = epochs
        except InvalidInput as exc:
            return {"error": str(exc)}, 400
        except FileNotFoundError as exc:            # 数据目录不存在 / 目录里没有可用数据文件
            return {"error": str(exc),
                    "hint": "检查 dataset_dir 与 dataset_type；CWRU 默认目录是 1DCNN/0HP"}, 409
        except ValueError as exc:                   # 未知模型名、rate 不合法、表里没有数值列、基线文件缺失…
            return {"error": str(exc)}, 400

        result = train(name, options)
        http = 200 if result["status"] == "成功" else 500
        if (result.get("db") or {}).get("written") is False:
            # 产物可能已经落盘、但库里没有 Trainings 行 —— 调用方必须知道，别只看 status
            result["warning"] = ("训练结果未写入数据库：" + str((result.get("db") or {}).get("error")))
        return result, http


class TrainingList(Resource):
    """GET /trainings?limit=N —— 最近训练记录（读库）。"""

    def get(self):
        limit = min(_int(request.args.get("limit"), 20, "limit") or 20, 200)
        try:
            return {"trainings": database.recent_trainings(limit), "dialect": database.dialect}
        except DBError as exc:
            return {"error": str(exc), "dialect": database.dialect}, 503


def _log_failed_inference(name: str, exc: Exception, client_ip: str | None, body: dict) -> None:
    """推理失败时也写一条 `ModelInvocations`（is_success=0）。

    以前只有成功路径写调用日志（而且写在结果插入之后），输入校验被拒、模型缺产物这些
    更常见的失败反而零痕迹，排查时只剩 Flask 的 500 页面。

    注意**不能**为了写日志去 `ensure_model`：模型名打错就会凭空多出一行 Models 登记。
    只有"该模型已有训练记录"（能拿到 ModelID 满足外键）时才写。
    """
    from .training import db_model_name                     # 懒导入，避免与 training 循环依赖
    try:
        row = database.latest_training(db_model_name(name), only_success=False)
        if not row:
            return
        if isinstance(exc, FileNotFoundError):
            code = 409
        elif isinstance(exc, (InvalidInput, ValueError)):
            code = 400
        else:
            code = 500
        database.insert_invocation(
            model_id=int(row["ModelID"]), training_id=int(row["TrainingID"]), api_endpoint="/predict",
            request_params={k: v for k, v in body.items() if k != "samples"},
            response_result={"error": f"{type(exc).__name__}: {exc}"},
            duration_ms=None, is_success=False, status_code=code, client_ip=client_ip, status="失败")
    except Exception:                                        # 写日志失败绝不能盖掉原始异常
        pass


class Predict(Resource):
    """POST /predict —— 推理入口。

    入参既可以是 samples（内联数组），也可以是 path（工作区内的文件）；
    成功返回结构化的 predictions + summary + figures + db 回执。
    """

    def post(self):
        body = _body()
        try:
            # 推理允许"上传进来的模型名"（不在 1dcnn/cwt_cnn/adtk 别名表里），
            # 所以这里不做严格校验，交给 predict() 去解析产物目录。
            name = (body.get("model") or "").strip()
            if not name:
                raise InvalidInput("model 必填")
            # 注意：不能写 `_int(...) or 默认值` —— 0 是 falsy，会把 limit=0 悄悄改成 1 放过去
            limit = _int(body.get("limit"), 1, "limit")
            limit = 1 if limit is None else limit
            if not (1 <= limit <= MAX_LIMIT):
                raise InvalidInput(f"limit 必须在 1..{MAX_LIMIT} 之间")
            # index 必须 ≥0：负索引在 Python 切片里是"从尾部数"，会切出错位窗口却照样通过校验
            index = _int(body.get("index"), 0, "index")
            index = 0 if index is None else index
            if index < 0:
                raise InvalidInput("index 不能为负数（窗口序号从 0 开始）")
            top_k = _int(body.get("top_k"), 3, "top_k")
            top_k = 3 if top_k is None else top_k
            if not (1 <= top_k <= 50):
                raise InvalidInput("top_k 必须在 1..50 之间")
            try:
                payload = predict(
                    model=name,
                    samples=body.get("samples"),
                    path=body.get("path"),
                    index=index,
                    limit=limit,
                    version=body.get("version"),
                    training_id=_int(body.get("training_id"), None, "training_id"),
                    top_k=top_k,
                    write_db=bool(body.get("write_db", True)),
                    client_ip=request.remote_addr,
                    column=body.get("column"),
                    sheet=body.get("sheet"),
                )
            except Exception as exc:                 # 失败也要留一条调用记录，再原样抛出
                _log_failed_inference(name, exc, request.remote_addr, body)
                raise
        except InvalidInput as exc:
            return {"error": str(exc)}, 400
        except FileNotFoundError as exc:
            return {"error": str(exc), "hint": "先调 POST /train 生成模型产物"}, 409
        except (DBError, ValueError) as exc:
            return {"error": str(exc)}, 400
        except Exception as exc:                                   # pragma: no cover
            return {"error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-1500:]}, 500
        # 推理成功但落库失败时，必须让调用方看得见（以前是静默 200，只有前端自己去看 db.written）
        if (payload.get("db") or {}).get("written") is False:
            payload["warning"] = ("推理已完成，但结果未写入数据库："
                                  + str((payload.get("db") or {}).get("error")))
        return payload, 200


class InferenceTaskList(Resource):
    """GET /inference-tasks?limit=N —— 最近推理任务（读库）。"""

    def get(self):
        limit = min(_int(request.args.get("limit"), 20, "limit") or 20, 200)
        try:
            return {"tasks": database.recent_inference_tasks(limit), "dialect": database.dialect}
        except DBError as exc:
            return {"error": str(exc), "dialect": database.dialect}, 503


class InferenceTaskDetail(Resource):
    """GET /inference-tasks/<id> —— 单个任务 + 它的 InferenceResults 明细。"""

    def get(self, task_id):
        try:
            task = database.inference_task(_int(task_id, None, "task_id"))
        except DBError as exc:
            return {"error": str(exc)}, 503
        if task is None:
            return {"error": f"InferenceTaskID={task_id} 不存在"}, 404
        return task


def _db_model_name(raw: str) -> str:
    """CRUD 场景允许任意**已登记**的模型名（新登记的不在 ALIASES 里）；简称照旧规范化。"""
    from .training import db_model_name
    try:
        return db_model_name(normalize_model(raw))
    except ValueError:
        return raw


def _artifact_key(raw: str) -> str:
    """把任意写法（1DCNN / 1dcnn / 上传的模型名）解析成 data/models 下的产物目录名。"""
    name = _db_model_name(raw)
    return next((k for k, v in MODEL_META.items() if v["db_name"].lower() == name.lower()), name.lower())


def _safe_model_name(name: str) -> str:
    """模型名的磁盘安全形式（与 /models/upload 里的处理保持一致）。"""
    safe = re.sub(r"[^\w\u4e00-\u9fa5.\-]+", "_", name).strip("._")
    if not safe:
        raise InvalidInput("模型名不合法（不能只有符号）")
    if safe != name:
        raise InvalidInput(f"模型名只能含中英文、数字、_ - .（收到 {name!r}）")
    return safe


def _rename_model(old: str, new: str) -> dict:
    """模型改名的**磁盘侧**动作：搬产物目录 + 改 meta.json + 改库里已存的路径。

    调用顺序（api 里就是这么用的）：先 _rename_model()，再 database.update_model(..., new_name=)；
    搬完目录后改库失败时用 _undo_rename() 搬回去，保证"库表"和"磁盘"不脱钩。
    """
    key = _artifact_key(old)                     # 真实目录名（1DCNN → 1dcnn）
    safe = _safe_model_name(new)
    old_dir, new_dir = config.model_dir / key, config.model_dir / safe
    moved = False
    if old_dir.is_dir():
        if new_dir.exists():
            raise InvalidInput(f"产物目录 data/models/{safe} 已存在，先删掉它或换个名字")
        old_dir.rename(new_dir)                  # 同一磁盘上的重命名，秒完成
        moved = True
        for meta_file in new_dir.glob("*/meta.json"):
            try:
                payload = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                # 目录里的 meta 都归这个模型，直接改写（上传时写的 model 大小写可能和目录名不一致，
                # 所以不做相等判断，免得 Windows 大小写不敏感时漏改）
                payload["model"] = safe
                meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        paths = database.rename_model_paths(key, safe)   # Trainings.ModelPath 等路径前缀
    except DBError:
        if moved:
            new_dir.rename(old_dir)
        raise
    return {"from": key, "to": safe, "artifact_dir_moved": moved,
            "artifact_dir": str(new_dir) if moved else None,
            "path_rows_updated": paths,
            "note": ("内置模型（1DCNN/cwt_cnn/adtk）改名后不能再按老名字训练——训练模板是硬编码的；"
                     "按新名字看档案/推理不受影响") if key in MODEL_META else None}


def _undo_rename(old: str, new: str) -> None:
    """update_model 失败时回滚 _rename_model 的副作用（目录 + meta + 路径）。"""
    key = _artifact_key(old)
    safe = re.sub(r"[^\w\u4e00-\u9fa5.\-]+", "_", new).strip("._") or new
    old_dir, new_dir = config.model_dir / key, config.model_dir / safe
    if new_dir.is_dir() and not old_dir.exists() and safe != key:
        try:
            new_dir.rename(old_dir)
        except OSError:
            return
    for meta_file in old_dir.glob("*/meta.json"):
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            payload["model"] = key
            meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        database.rename_model_paths(safe, key)
    except DBError:
        pass


class ArtifactDetail(Resource):
    """GET /models/<model_name> —— 查看某个模型产物的 meta.json；DELETE 见下。"""

    def get(self, model_name):
        """取产物档案。model_name 允许任意写法（1DCNN / 1dcnn / 上传名），先归一再查。"""
        try:
            artifact = load_artifact(_artifact_key(model_name), request.args.get("version"))
        except FileNotFoundError as exc:
            return {"error": str(exc)}, 404
        except ValueError as exc:
            return {"error": str(exc)}, 400
        return artifact.to_dict()

    def delete(self, model_name):
        """删除：?version=vN 删产物版本；?scope=record 删 Models 表登记行（带引用检查）。"""
        try:
            version = request.args.get("version")
            scope = (request.args.get("scope") or "").strip()
            if version:
                return delete_version(_artifact_key(model_name), version), 200
            if scope == "record":
                force = request.args.get("force") in ("1", "true", "True")
                return database.delete_model(_db_model_name(model_name), force=force), 200
            raise InvalidInput("删除必须指定 ?version=vN（删产物版本）或 ?scope=record（删库表登记行）")
        except FileNotFoundError as exc:
            return {"error": str(exc)}, 404
        except DBError as exc:
            return {"error": str(exc)}, 409
        except (ValueError, InvalidInput) as exc:
            return {"error": str(exc)}, 400

    def put(self, model_name):
        """改 Models 表的登记信息：Description / ModelType / ApiEndpoint / Status / IsActive。

        带 `NewModelName`（或 `new_name`）时**连产物目录一起改名**：
        `data/models/<老名>` → `data/models/<新名>`，同时改产物 meta.json 里的 `model` 字段、
        以及 Trainings.ModelPath / InferenceTasks.InputPath 等已落库的路径前缀，
        避免出现"库表改名了、磁盘还是老目录"的脱钩状态。
        """
        body = _body()
        old = _db_model_name(model_name)
        new_name = str(body.get("NewModelName") or body.get("new_name") or "").strip()
        rename = None
        try:
            if new_name and new_name != old:
                if database.model_exists(new_name):
                    return {"error": f"模型名 {new_name} 已被占用，换一个"}, 409
                rename = _rename_model(old, new_name)
            try:
                result = database.update_model(old, body, new_name=new_name or None)
            except Exception:
                if rename:
                    _undo_rename(old, new_name)           # 搬完目录但改库失败 → 原样搬回去
                raise
        except FileNotFoundError as exc:
            return {"error": str(exc)}, 404
        except DBError as exc:
            return {"error": str(exc)}, 409
        except (ValueError, InvalidInput) as exc:
            return {"error": str(exc)}, 400
        if rename:
            result["rename"] = rename
        return result, 200


class ModelOverview(Resource):
    """一个模型的完整档案：登记信息 + 产物参数(meta) + 最近训练 + 引用统计。

    模型列表要同时显示「登记信息」与「模型参数」，并且选中某行要展开它自己的参数——
    分散在 4 个地方（Models 表 / Trainings 表 / 产物 meta.json / 引用关系），
    这里合并成一个响应，前端一次请求即可，不用自己拼。
    """

    def get(self, model_name):
        """一个模型的完整档案：登记 + 产物参数 + 最近训练 + 引用统计，一次请求给全。"""
        name = _db_model_name(model_name)
        key = next((k for k, v in MODEL_META.items() if v["db_name"].lower() == name.lower()), name.lower())

        registration = next((r for r in database.models_in_db()
                             if str(r["ModelName"]).lower() == name.lower()), None)
        versions = [a.to_dict() for a in list_artifacts(key)]
        params, metrics, labels, dataset, confusion = {}, {}, None, None, None
        if versions:
            meta = list_artifacts(key)[-1].meta          # 最新版本
            params = meta.get("params") or {}
            metrics = meta.get("metrics") or {}
            labels = meta.get("labels")
            dataset = meta.get("dataset")
            confusion = meta.get("confusion")
        trainings = [t for t in database.recent_trainings(50)
                     if str(t.get("ModelName", "")).lower() == name.lower()][:5]
        try:
            references = database.model_references(name)
        except DBError as exc:
            references = {"error": str(exc)}
        return {
            "model": name, "artifact_key": key,
            "registration": registration,
            "artifact_versions": versions,
            "latest_version": versions[-1]["version"] if versions else None,
            "params": params, "metrics": {k: v for k, v in metrics.items() if k != "history"},
            "labels": labels, "dataset": dataset, "confusion": confusion,
            "recent_trainings": trainings,
            "references": references,
        }, 200


class ModelUpload(Resource):
    """上传一个**模型（文件夹，或者单个/多个文件）**，落盘成 data/models/<模型名>/<版本>/ 并登记 Models 表。

    表单字段：`name`（模型名，必填）、`version`（可选，默认自动取下一个 vN）、
    以及多个 `file`——浏览器既可以用 `<input type="file" webkitdirectory>` 选整个文件夹，
    也可以用普通 `<input type="file" multiple>` 选单个/多个文件，两条路走同一套逻辑
    （服务端按 basename 扁平化保存，丢掉相对路径）。

    **不再要求填 input_len / num_classes / labels**：
      1. 先按扩展名挑出权重候选，再用 probe_weight() 逐个做**内容探测**，判断"这是不是一个模型"；
         一个都不通过 → 400，并把每个文件被拒的原因一起返回（前端直接展示）；
      2. 通过的即权重文件，input_len / 类别数尽量从文件里读出来（Keras 解析 model_config，
         PyTorch 读最后一个全连接层的 weight 形状）；
      3. 文件夹里带 meta.json 就以它为准，只把缺失的 input_len/num_classes 补上。
    可选 scaler.npz；实在读不出 input_len 时 meta 里留 null，并在响应 warnings 里说明
    （推理侧按默认 784 处理，想固定就手动补 meta.json）。
    """

    WEIGHT_WHITELIST = WEIGHT_SUFFIXES
    KEEP_SUFFIX = KEEP_SUFFIXES
    MAX_MB = 500

    def post(self):
        """接收文件夹或若干文件，探测→落盘→登记 Models 表。"""
        import shutil
        import numpy as np
        from .registry import _VERSION_RE, next_version_dir

        name = (request.form.get("name") or "").strip()
        files = request.files.getlist("file") or request.files.getlist("files")
        if not name:
            return {"error": "name（模型名）必填"}, 400
        if not files:
            return {"error": "没有收到文件（表单字段名用 file，可多选/整个文件夹）"}, 400
        safe = re.sub(r"[^\w\u4e00-\u9fa5.\-]+", "_", name).strip("._") or "model"

        # 1) 先把文件读进内存并分类，避免半途落盘
        blobs: dict[str, bytes] = {}
        skipped, weights, scaler, meta_blob = [], None, None, None
        candidates: list[tuple[str, str]] = []          # 权重候选：(文件名, 框架)，按上传顺序
        for item in files:
            filename = Path((item.filename or "").replace("\\", "/")).name
            if not filename or filename.startswith("."):
                continue
            suffix = Path(filename).suffix.lower()
            if suffix not in self.KEEP_SUFFIX:
                skipped.append({"filename": filename, "reason": f"忽略非模型文件（{suffix or '无扩展名'}）"})
                continue
            blob = item.read()
            if len(blob) > self.MAX_MB * 1024 * 1024:
                skipped.append({"filename": filename, "reason": f"超过 {self.MAX_MB}MB"})
                continue
            blobs[filename] = blob
            if suffix in self.WEIGHT_WHITELIST:
                candidates.append((filename, self.WEIGHT_WHITELIST[suffix]))
            if filename == "scaler.npz":
                scaler = filename
            if filename == "meta.json":
                meta_blob = blob

        # 1.5) 判断"这是不是一个模型"：候选逐个做内容探测，第一个通过的当权重
        if not candidates:
            return {"error": "上传的内容里没有权重文件，不算模型（支持 .h5/.keras/.pt/.pth/.pkl/.pickle）",
                    "received": sorted(blobs), "skipped": skipped}, 400
        probe: dict = {}
        probed: list[dict] = []
        for filename, framework in candidates:
            info = probe_weight(filename, blobs[filename])
            probed.append({"filename": filename, "framework": framework, "ok": info["ok"],
                           "reason": info["reason"], "input_len": info["input_len"],
                           "num_classes": info["num_classes"]})
            if info["ok"]:
                weights, probe = (filename, framework), info
                break
        if weights is None:
            return {"error": "上传的内容不是一个模型：没有任何文件通过权重校验",
                    "detail": probed, "skipped": skipped,
                    "hint": "支持 Keras 的 .h5/.keras（需含 model_weights 组或 config.json）、"
                            "PyTorch 的 .pt/.pth（torch 的 zip 检查点）、pickle 的 .pkl"}, 400

        # 2) 决定版本目录
        version = (request.form.get("version") or "").strip()
        root = config.model_dir / safe
        if version:
            if not _VERSION_RE.match(version):
                return {"error": f"版本号格式不合法：{version}（应为 v1、v2 这种）"}, 400
            target = root / version
            if target.exists():
                return {"error": f"{safe}/{version} 已存在，换个版本号或先删除该版本"}, 409
            target.mkdir(parents=True, exist_ok=False)
        else:
            target = next_version_dir(safe)

        # 3) 落盘
        try:
            for filename, blob in blobs.items():
                (target / filename).write_bytes(blob)
            meta = None
            if meta_blob:
                try:
                    meta = json.loads(meta_blob.decode("utf-8"))
                except Exception:
                    meta = None
            if not meta:      # 缺 meta.json → 用探测结果自己生成（表单里填了就以表单为准）
                input_len = probe.get("input_len") or _int(request.form.get("input_len"), None, "input_len")
                num_classes = probe.get("num_classes") or _int(request.form.get("num_classes"), None, "num_classes")
                labels = [s.strip() for s in (request.form.get("labels") or "").split(",") if s.strip()]
                if not labels and num_classes == 10:      # CWRU 十类，标签直接给现成的
                    labels = [label for _, _, label in sorted(ds.CWRU_0HP_CLASSES, key=lambda row: row[1])]
                meta = {
                    "model": safe, "version": target.name, "framework": weights[1],
                    "task": "classification" if weights[1] != "adtk" else "anomaly_detection",
                    "input_len": input_len, "num_classes": num_classes or (len(labels) or None),
                    "labels": labels or None, "weights_file": weights[0],
                    "scaler_file": scaler if scaler else (Path("scaler.npz").name if (target / "scaler.npz").is_file() else None),
                    "params": {"source": "uploaded"}, "metrics": {},
                    "dataset": {"name": request.form.get("dataset") or None, "path": None, "stats": {}},
                    "trained_at": None, "uploaded_at": datetime.now().isoformat(timespec="seconds"),
                    "origin": "POST /models/upload",
                    "trusted": False,            # 上传的产物一律视为不可信：推理侧不会反序列化它的 .pkl
                    "provenance": "upload",
                    "probe": {"weights": weights[0], "framework": weights[1],
                              "input_len": probe.get("input_len"), "num_classes": probe.get("num_classes"),
                              "reason": probe.get("reason")},
                }
                (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                meta.setdefault("version", target.name)
                meta.setdefault("weights_file", weights[0])
                # 上传的 meta.json 是用户提供的，它自称 trusted 也不算数：强制标成不可信
                meta["trusted"] = False
                meta["provenance"] = "upload"
                for key in ("input_len", "num_classes"):   # meta 里缺的，用探测结果补齐
                    if not meta.get(key) and probe.get(key):
                        meta[key] = probe[key]
                if scaler:
                    meta.setdefault("scaler_file", scaler)
                (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            shutil.rmtree(target, ignore_errors=True)
            return {"error": f"落盘失败，已回滚：{type(exc).__name__}: {exc}"}, 500

        # 4) 登记 Models 表（同名就取用，不重复插）
        model_type = {"classification": "Classification", "anomaly_detection": "AnomalyDetection",
                      "regression": "Regression"}.get(str(meta.get("task") or "").lower(), meta.get("task"))
        try:
            model_id = database.ensure_model(safe, description=request.form.get("description"),
                                             model_type=model_type, status="可运行")
        except DBError as exc:
            model_id, db_error = None, str(exc)
        else:
            db_error = None

        warnings = []
        if not meta.get("input_len"):
            warnings.append("没能自动识别输入长度（input_len），推理时按默认 784 处理；"
                            "想固定长度就在产物目录的 meta.json 里补一个 input_len")
        if meta.get("weights_file") and meta["weights_file"] != weights[0]:
            warnings.append(f"meta.json 声明的权重是 {meta['weights_file']}，探测选中的是 {weights[0]}（以 meta 为准）")

        return {
            "model": safe, "version": target.name, "directory": str(target),
            "framework": meta.get("framework"), "weights": meta.get("weights_file") or weights[0],
            "input_len": meta.get("input_len"), "num_classes": meta.get("num_classes"),
            "labels": meta.get("labels"), "scaler": meta.get("scaler_file"),
            "meta_generated": not bool(meta_blob),
            "probe": {"weights": weights[0], "framework": weights[1], "reason": probe.get("reason"),
                      "input_len": probe.get("input_len"), "num_classes": probe.get("num_classes")},
            "probed_files": probed,
            "saved": [{"filename": k, "size_kb": round(len(v) / 1024, 1)} for k, v in blobs.items()],
            "skipped": skipped, "warnings": warnings,
            "db": {"written": db_error is None, "ModelID": model_id, "ModelName": safe, "error": db_error},
            "hint": f"现在可以在「推理」里选 {safe}（输入长度 {meta.get('input_len') or '默认 784'}）",
        }, 201


class ModelCreate(Resource):
    """POST /models —— 新增一条 Models 表登记（不训练，只是登记/占位）。"""

    def post(self):
        body = _body()
        name = (body.get("ModelName") or body.get("name") or "").strip()
        if not name:
            return {"error": "ModelName 不能为空"}, 400
        try:
            model_id = database.ensure_model(
                name, description=body.get("Description"), model_type=body.get("ModelType"),
                api_endpoint=body.get("ApiEndpoint") or "/predict", status=body.get("Status") or "未训练")
        except DBError as exc:
            return {"error": str(exc)}, 409
        return {"ModelID": model_id, "ModelName": name}, 201


class ModelReferences(Resource):
    """GET /models/<model_name>/references —— 删前引用体检：被哪些表引用了多少行。"""

    def get(self, model_name):
        try:
            return database.model_references(_db_model_name(model_name)), 200
        except DBError as exc:
            return {"error": str(exc)}, 404


class DatasetRecord(Resource):
    """GET/PUT/DELETE /datasets/db/<dataset_id> —— 单条 Datasets 记录（查/改/删）。"""

    def put(self, dataset_id):
        try:
            return database.update_dataset(int(dataset_id), _body()), 200
        except DBError as exc:
            return {"error": str(exc)}, 409

    def delete(self, dataset_id):
        """删登记行；被 Trainings/InferenceTasks 引用时默认拒绝，?force=true 才连带清理。"""
        force = request.args.get("force") in ("1", "true", "True")
        try:
            return database.delete_dataset(int(dataset_id), force=force), 200
        except DBError as exc:
            return {"error": str(exc)}, 409

    def get(self, dataset_id):
        """取单条 Datasets 记录（没有按主键查的接口，这里在列表里找）。"""
        for row in database.datasets_in_db():
            if int(row["DatasetID"]) == int(dataset_id):
                return row, 200
        return {"error": f"DatasetID={dataset_id} 不存在"}, 404


class FigureList(Resource):
    """GET /figures —— 列出已生成的图（训练曲线/混淆矩阵/预测分布等）。"""

    def get(self):
        # 先按大 limit 取回来再按 model 过滤，避免"过滤后不足 limit 条"这种别扭语义
        limit = min(_int(request.args.get("limit"), 60, "limit") or 60, 500)
        model = request.args.get("model")
        items = list_figures(limit=500)
        if model:
            items = [i for i in items if i["file"].startswith(f"{model}/") or f"/{model}/" in i["file"]]
        return {"count": len(items[:limit]), "figures": items[:limit],
                "hint": "图片可直接用返回的 url 在浏览器打开（GET /figures/<路径>）"}


class FigureFile(Resource):
    """GET /figures/<路径> —— 直接返回 PNG 文件（conditional=True 支持 304 缓存协商）。"""

    def get(self, relpath):
        return send_from_directory(FIG_DIR, relpath, conditional=True)


class Console(Resource):
    """GET /ui —— 零构建单页控制台：一个 HTML（内联 CSS/JS）直接由 Flask 吐出，不需要 npm/打包。

    路径用 /ui 而不是 /console：Flask 开 debug=True 时，Werkzeug 的调试器会独占
    `/console`（浏览器打开会是它的 "Confirm Pin" 页面，而不是我们的控制台）。
    """

    def get(self):
        # no-store：控制台改完刷新就能看到，不用手动清缓存
        html = CONSOLE_HTML.read_text(encoding="utf-8")
        return Response(html, mimetype="text/html", headers={"Cache-Control": "no-store"})


class Root(Resource):
    """GET / —— 根路径直接把人送到控制台，避免 127.0.0.1:5000/ 看到 404。"""

    def get(self):
        return redirect("/ui")


# ============================ 数据集管理 / 数据展示 ============================
class DatasetDb(Resource):
    """GET/POST /datasets/db —— Datasets 表的登记记录：读 + 登记。"""

    def get(self):
        """列登记记录，顺带把 8 张表的行数一并返回（前端「库表登记」页要用）。"""
        try:
            return {"datasets": database.datasets_in_db(), "dialect": database.dialect,
                    "counts": database.table_counts()}, 200
        except DBError as exc:
            return {"error": str(exc), "dialect": database.dialect}, 503

    def post(self):
        """登记一个数据集（同名则沿用已有行，响应里的 already_existed 标明是哪种）。"""
        body = _body()
        name = (body.get("name") or "").strip()
        if not name:
            return {"error": "name 不能为空"}, 400
        try:
            result = database.register_dataset(
                name=name, source=body.get("source"),
                class_count=_int(body.get("class_count"), None, "class_count"),
                sample_count=_int(body.get("sample_count"), None, "sample_count"),
                data_path=body.get("data_path"), description=body.get("description"))
        except DBError as exc:
            return {"error": str(exc), "dialect": database.dialect}, 503
        result["dialect"] = database.dialect
        return result, 200 if result["already_existed"] else 201


class DatasetSignal(Resource):
    """取一段原始信号（降采样成数值数组），供「数据展示」画波形。

    既支持内置 .mat（CWRU DE 通道），也支持表格文件（.csv/.xlsx/.xls，可指定 column/sheet）。
    """

    def get(self):
        """取一段原始信号（已降采样成 points 个数值）。"""
        dataset = request.args.get("dataset") or ""
        filename = request.args.get("file")
        if not filename:
            return {"error": "缺少 file 参数（数据集文件名）"}, 400
        points = min(max(_int(request.args.get("points"), 1500, "points") or 1500, 200), 4000)
        start = max(_int(request.args.get("start"), 0, "start") or 0, 0)
        column = request.args.get("column")
        sheet = request.args.get("sheet")

        # 1) 目录来自 dataset 键（内置 CWRU / 上传的表格数据集）
        directory = None
        if dataset in config.dataset_dirs:
            directory = config.dataset_dirs[dataset]
        elif dataset.startswith("表格:"):
            directory = config.upload_dir / dataset.split(":", 1)[1]
        elif dataset:
            directory = config.upload_dir / dataset
        if directory is not None:
            try:
                # 数据集目录必须落在工作区内：否则 dataset=..\..\..\x 能读到工作区外的文件
                # （原来的写法只校验了"逃逸之后的目录"，等于没校验）
                directory.resolve().relative_to(config.workspace_dir.resolve())
            except ValueError:
                return {"error": f"非法的 dataset 参数：{dataset!r}（越出工作区）"}, 400
            path = (directory / Path(filename).name).resolve()
            try:
                path.relative_to(directory.resolve())
            except ValueError:
                return {"error": f"文件不在该数据集目录内：{filename}"}, 400
        else:                                   # 2) 直接给工作区内的相对/绝对路径
            try:
                path = _resolve_workspace_path(filename)
            except InvalidInput as exc:
                return {"error": str(exc)}, 400
        if not path.is_file():
            return {"error": f"文件不存在：{path.name}"}, 404

        try:
            if tabular.is_table(path):
                signal = tabular.read_signal(path, column=column, sheet=sheet)
                label = tabular.label_from_filename(path.name)
                class_id = None
                kind = "tabular"
            else:
                signal = ds.read_de_channel(path)
                label, class_id, kind = ds.guess_label(path.name), None, "matlab"
                for fname, cid, lab in ds.CWRU_0HP_CLASSES:
                    if fname == path.name:
                        label, class_id = lab, cid
                        break
        except Exception as exc:
            return {"error": f"读取信号失败：{type(exc).__name__}: {exc}"}, 500

        segment = signal[start:]
        stride = max(1, segment.size // points)
        values = segment[::stride][:points]
        return {
            "dataset": dataset or None, "file": path.name, "file_path": str(path),
            "dataset_type": kind, "label": label, "class_id": class_id, "column": column, "sheet": sheet,
            "samples_in_file": int(signal.size), "start": start, "stride": stride,
            "points": int(values.size),
            "min": round(float(values.min()), 5), "max": round(float(values.max()), 5),
            "mean": round(float(values.mean()), 5), "std": round(float(values.std()), 5),
            "values": [round(float(v), 5) for v in values],
        }, 200


# ================================ 系统管理 ================================
class SystemInfo(Resource):
    """GET /system —— 运行信息：Python/平台、关键依赖版本、路径、库表行数、产物/图/日志 占用。"""

    def get(self):
        def dist_version(name):
            """查已安装包的版本；没装返回 None（前端显示"未安装"）。"""
            try:
                return version(name)
            except PackageNotFoundError:
                return None
            except Exception:                                            # pragma: no cover
                return "读取失败"

        artifacts = list_artifacts()
        figure_items = list_figures(limit=1000)
        logs = list(config.log_dir.glob("*.log"))
        try:
            counts, db_ok = database.table_counts(), True
        except DBError as exc:
            counts, db_ok = {"error": str(exc)}, False
        return {
            "runtime": {"python": sys.version.split()[0], "executable": sys.executable,
                        "platform": platform.platform(), "machine": platform.machine()},
            "packages": {name: dist_version(name) for name in _PACKAGES},
            "paths": config.describe(),
            "database": {"dialect": database.dialect, "ok": db_ok, "counts": counts,
                         "bootstrap": database.last_bootstrap},
            "artifacts": {"count": len(artifacts),
                          "items": [{"model": a.name, "version": a.version, "framework": a.framework,
                                     "weights": str(a.weights),
                                     "size_kb": round(a.weights.stat().st_size / 1024, 1)} for a in artifacts]},
            "figures": {"count": len(figure_items),
                        "total_kb": round(sum(f["size_kb"] for f in figure_items), 1), "dir": str(FIG_DIR)},
            "logs": {"count": len(logs),
                     "total_kb": round(sum(p.stat().st_size for p in logs) / 1024, 1),
                     "dir": str(config.log_dir)},
        }, 200


class SystemLogs(Resource):
    """GET /system/logs —— 训练日志文件列表（按修改时间倒序）。"""

    def get(self):
        files = sorted(config.log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return {"dir": str(config.log_dir), "count": len(files), "logs": [
            {"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1),
             "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
            for p in files]}, 200


class SystemLogFile(Resource):
    """GET /system/logs/<name>?tail=N —— 看某个日志的尾部 N 行。"""

    def get(self, name):
        # 只接受纯文件名：挡掉 `../` 之类的路径，否则能读到日志目录之外的文件
        if Path(name).name != name:
            return {"error": "非法日志名"}, 400
        path = config.log_dir / name
        if not path.is_file():
            return {"error": f"日志不存在：{name}"}, 404
        tail = min(max(_int(request.args.get("tail"), 300, "tail") or 300, 10), 3000)
        # Keras 进度条写进日志的 ANSI 转义序列在浏览器里是乱码，这里统一剥掉
        lines = [_ANSI.sub("", ln).rstrip()
                 for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
        return {"name": name, "size_kb": round(path.stat().st_size / 1024, 1),
                "total_lines": len(lines), "returned": len(lines[-tail:]), "lines": lines[-tail:]}, 200


class Maintenance(Resource):
    """POST /system/maintenance —— 维护操作（危险，前端红按钮 + 二次确认）。"""

    def post(self):
        target = (_body().get("target") or "").strip()
        if target == "figures":
            return clear_figures(), 200
        # 白名单式：只认已知的 target，别的明确报错并回可选值
        return {"error": f"不支持的维护目标：{target!r}", "supported": ["figures"]}, 400


def register_api(api) -> None:
    """把资源挂到 flask_restful.Api 上（由 main.py 调用）。"""
    api.add_resource(Root, "/")
    api.add_resource(Console, "/ui")
    api.add_resource(ApiIndex, "/api")
    api.add_resource(Health, "/health")
    api.add_resource(ModelList, "/models")
    api.add_resource(DatasetList, "/datasets")
    api.add_resource(Train, "/train")
    api.add_resource(TrainingList, "/trainings")
    api.add_resource(Predict, "/predict")
    api.add_resource(InferenceTaskList, "/inference-tasks")
    api.add_resource(InferenceTaskDetail, "/inference-tasks/<int:task_id>")
    api.add_resource(ArtifactDetail, "/models/<model_name>")
    api.add_resource(ModelCreate, "/models")
    api.add_resource(ModelUpload, "/models/upload")
    api.add_resource(ModelReferences, "/models/<model_name>/references")
    api.add_resource(ModelOverview, "/models/<model_name>/overview")
    api.add_resource(DatasetRecord, "/datasets/db/<int:dataset_id>")
    api.add_resource(FigureList, "/figures")
    api.add_resource(FigureFile, "/figures/<path:relpath>")
    # 数据集管理 / 数据展示
    api.add_resource(DatasetDb, "/datasets/db")
    api.add_resource(DatasetSignal, "/datasets/signal")
    api.add_resource(DatasetUpload, "/datasets/upload")
    api.add_resource(TablePreview, "/datasets/table")
    # 系统管理
    api.add_resource(SystemInfo, "/system")
    api.add_resource(SystemLogs, "/system/logs")
    api.add_resource(SystemLogFile, "/system/logs/<name>")
    api.add_resource(Maintenance, "/system/maintenance")

    _install_path_mask(api)          # 出口脱敏：响应里不再出现本机真实目录


# ============================================ 响应脱敏：不把本机真实目录暴露给前端
_DRIVE_RE = re.compile(r"[A-Za-z]:[\\/]")


def mask_private_paths(payload, workspace: str = "", home: str = ""):
    """递归把响应里的绝对路径换成"工作区内的相对路径"。

    路径散落在 30 多个字段里（`directory` / `weights` / `dataset.path` / `log_file` /
    `figures_dir` / `input_path` / `output_path` / 报错信息里的路径 …），逐个改必然漏，
    所以在**响应出口**统一兜一遍：

        D:\\22project\\testRestfulProject\\data\\models\\1dcnn\\v1
            → testRestfulProject\\data\\models\\1dcnn\\v1
        C:\\Users\\<用户名>\\...   → <用户目录>\\...
        E:\\其它盘\\...           → <本机>\\...

    注意只影响**显示**：前端要回传的路径（如 `dataset_dir`）变成"相对工作区"后，
    后端用 `project_dir / 相对路径` 仍能解析，训练/推理链路不受影响。
    """
    def fix(text: str) -> str:
        """把一段文本里出现的本机绝对路径换成占位符标签。"""
        if workspace and text == workspace:
            return "<工作区>"
        if home and text == home:
            return "<用户目录>"
        for raw, label in ((workspace, ""), (home, "<用户目录>")):
            if raw:
                text = text.replace(raw + "\\", label).replace(raw + "/", label).replace(raw, label)
        # 用 lambda 而不是字符串模板：替换串以反斜杠结尾会让 re.sub 抛 "bad escape (end of pattern)"
        return _DRIVE_RE.sub(lambda _match: "<本机>/", text)

    def walk(node):
        """递归遍历 JSON 结构，对每个字符串套 fix()；容器原样重建，其它类型不动。"""
        if isinstance(node, str):
            return fix(node)
        if isinstance(node, dict):
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(item) for item in node]
        return node

    return walk(payload)


def _install_path_mask(api) -> None:
    """给 Flask app 挂一个 after_request：只处理 JSON 响应，图片/日志/SSE 原样放行。"""
    app = getattr(api, "app", None)
    if app is None:                                              # pragma: no cover
        return
    workspace, home = str(config.workspace_dir), str(Path.home())

    @app.after_request
    def _mask_response(response):                                # noqa: ANN001
        """出口统一兜一遍脱敏，避免逐个字段改还漏掉。"""
        mimetype = response.mimetype or ""
        if mimetype == "application/json":
            try:
                payload = json.loads(response.get_data(as_text=True) or "null")
            except Exception:                                    # 不是合法 JSON 就别动它
                return response
            response.set_data(json.dumps(mask_private_paths(payload, workspace, home), ensure_ascii=False))
        elif mimetype.startswith("text/") and mimetype != "text/event-stream":
            # 日志尾部（/system/logs/<名>）里也有绝对路径，一起脱敏
            response.set_data(mask_private_paths(response.get_data(as_text=True), workspace, home))
        return response
