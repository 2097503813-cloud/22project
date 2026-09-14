# -*- coding: utf-8 -*-
"""模型产物（流程图里的「Pxl模型」盒子）。

原来两个训练脚本跑完就把模型丢在内存里，没有 model.save / torch.save，
所以「Pxl模型」这个盒子在磁盘上根本不存在，Trainings.ModelPath 也无值可填。
本模块定义一套极简的产物约定：

    data/models/<模型名>/v1/model.keras      ← Keras 原生格式
    data/models/<模型名>/v1/model.pt         ← PyTorch state_dict
    data/models/<模型名>/v1/detector.pkl     ← 无监督检测器（adtk 的 PcaAD）
    data/models/<模型名>/v1/meta.json        ← 输入长度、类别表、指标、超参、数据集指纹

meta.json 一定要带类别表：否则模型文件本身无法解释 0..9 到底对应哪种故障。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import config

_WEIGHT_NAMES = ("model.keras", "model.pt", "detector.pkl", "model.h5")
_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass
class Artifact:
    """一个已落盘的模型产物：**定位信息**（目录 / 权重文件）+ **自解释信息**（框架 / meta）。

    它不是"模型对象"，只是磁盘上那堆文件的**句柄**——推理时按 `weights` 加载、
    按 `framework` 决定用哪个引擎、按 `meta["input_len"]` 决定切多长的窗。
    """

    name: str                 # 模型名（同时也是 data/models 下的目录名，如 1dcnn）
    version: str              # 版本目录名，形如 v1 / v2
    directory: Path           # 版本目录的绝对路径 data/models/<名>/<版本>/
    weights: Path             # 权重文件绝对路径（model.h5 / model.pt / detector.pkl）
    framework: str            # tensorflow-keras / pytorch / adtk —— 推理分派靠它
    meta: dict = field(default_factory=dict)   # meta.json 的完整内容（见模块头）

    def to_dict(self) -> dict:
        """挑出给接口/前端用的字段（meta 里的原始 dict 太大，不透传）。

        只挑 12 个字段是有意为之：`/models` 会一次列出所有模型的全部版本，
        把整份 meta（含 history 曲线、classification_report 文本）塞进去会让响应膨胀几十倍。
        """
        return {
            "model": self.name,
            "version": self.version,
            "framework": self.framework,
            "weights": str(self.weights),
            "directory": str(self.directory),
            "input_len": self.meta.get("input_len"),
            "num_classes": self.meta.get("num_classes"),
            "labels": self.meta.get("labels"),
            "metrics": self.meta.get("metrics"),
            "params": self.meta.get("params"),
            "dataset": self.meta.get("dataset"),
            "created_at": self.meta.get("created_at"),
        }


# \u6a21\u578b\u540d\u5141\u8bb8\u7684\u5b57\u7b26\uff1a\u4e2d\u82f1\u6587\u3001\u6570\u5b57\u3001\u4e0b\u5212\u7ebf\u3001\u70b9\u3001\u6a2a\u7ebf\uff08\u89c1 _model_root \u7684\u5b89\u5168\u8bf4\u660e\uff09
_MODEL_NAME_RE = re.compile(r"^[\w\u4e00-\u9fa5.\-]+$")


def _model_root(name: str) -> Path:
    """模型名 → 产物目录。**必须净化**。

    模型名既可能来自别名表，也可能来自"上传的任意模型名"，而它会被直接拼进路径。
    不净化的话 `../../x` 能读到工作区外的目录，推理时还会对这些文件做反序列化 ——
    等于给了一个任意文件读取 / 任意代码执行的入口。
    只允许中英文、数字、下划线、点、横线，且不含 `..`。
    """
    clean = str(name or "").strip()
    if not clean or ".." in clean or not _MODEL_NAME_RE.match(clean):
        raise ValueError(f"非法的模型名 {name!r}：只允许中英文、数字、_ - .，且不能含路径分隔符")
    return config.model_dir / clean


def next_version_dir(name: str) -> Path:
    """算出并**创建**下一个版本目录，形如 data/models/1dcnn/v2。

    编号规则：取现有 vN 里最大的 N 再加 1（删掉 v1 后不会复用编号，避免"同名不同物"）。
    这里 `mkdir(exist_ok=False)` 是并发的安全网：两个训练请求同时进来时，
    后一个会撞 FileExistsError 而不是写进同一个目录把产物搅坏。
    """
    root = _model_root(name)
    root.mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for p in root.iterdir() if p.is_dir() and (m := _VERSION_RE.match(p.name))]
    version = f"v{max(used, default=0) + 1}"
    target = root / version
    target.mkdir(parents=True, exist_ok=False)   # 目录已存在就抛错，不做静默覆盖
    return target


def _read_meta(directory: Path) -> dict:
    """读 meta.json；缺失或内容坏掉都返回空 dict，让调用方走默认分支而不是崩掉。"""
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _find_weights(directory: Path) -> Path | None:
    """找权重文件。

    优先用 meta.json 里登记的 weights_file——因为发生过「.keras 保存失败但留下
    只含 config.json 的半成品 zip，被固定顺序误当成权重」的事故；
    其次按约定顺序找，并且忽略 0 字节的空壳。
    """
    recorded = _read_meta(directory).get("weights_file")
    if recorded:
        candidate = directory / recorded
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    for name in _WEIGHT_NAMES:
        p = directory / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def save_artifact(name: str, framework: str, saver, meta: dict) -> Artifact:
    """落盘一个模型产物，返回可用的 Artifact。

    saver: 调用方提供的回调，签名 `saver(target_dir: Path) -> Path`，负责把权重写进
    刚建好的版本目录并返回权重文件路径（各框架的存法不同，所以由 trainer 自己决定）。

    整个落盘是一个"要么全有要么全无"的单元：中途任何异常都会把刚建的版本目录
    **整个删掉**（`rmtree`），避免留下"只有 scaler 没有权重"的半成品被后续查找误命中。
    ⚠️ 原来的 `keep_previous=False` 分支（外加它专用的 `_prune_except()`）已删除：
    全项目没有一处调用方传过这个参数，那个分支永远走不到，历史版本本来就是一律保留的。
    """
    target = next_version_dir(name)
    try:
        # ① 先让 saver 写权重（Keras 的 .keras / PyTorch 的 .pt / adtk 的 pickle 都在这步）
        weights = saver(target)
        # ② meta 是产物的"说明书"：模型名/版本/框架 + 权重文件名，再合并 trainer 给的业务字段
        #    （input_len、labels、metrics、params、dataset 指纹…）。权重文件名必须记下来，
        #    否则后面只能靠固定顺序猜哪个文件是权重（踩过 .keras 半成品被误命中的坑）。
        meta = {
            **meta,
            "model": name,
            "version": target.name,
            "framework": framework,
            "weights_file": Path(weights).name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return Artifact(name=name, version=target.name, directory=target,
                        weights=Path(weights), framework=framework, meta=meta)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)   # 回滚：不留半个产物
        raise


def load_artifact(name: str, version: str | None = None) -> Artifact:
    """按名字取产物，`version` 为空时取**最新版**（推理默认走这里）。

    抛错语义（上层据此回 404/409，不要吞）：
      - 模型目录不存在 / 一个版本都没有 → FileNotFoundError("先调 /train")
      - 指定版本不存在               → FileNotFoundError
      - 版本目录在但找不到权重       → FileNotFoundError（半成品目录会走到这一支）
    """
    root = _model_root(name)
    if not root.is_dir():
        raise FileNotFoundError(f"还没有任何 {name} 的模型产物，先调 /train")
    if version:
        directory = root / version
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} 不存在版本 {version}")
    else:
        # 版本目录按数字排序取最大者（不能按字符串排：v10 会排在 v2 前面）
        candidates = sorted([p for p in root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)],
                            key=lambda p: int(p.name[1:]))
        if not candidates:
            raise FileNotFoundError(f"还没有任何 {name} 的模型产物，先调 /train")
        directory = candidates[-1]

    weights = _find_weights(directory)
    if weights is None:
        raise FileNotFoundError(f"{directory} 里找不到权重文件")
    meta_path = directory / "meta.json"
    # meta 缺失也放行（framework 退化成 unknown），让"权重存在但没有说明书"的产物还能被看到，
    # 而不是在这里直接崩掉——真正需要 input_len/labels 的地方自己会报错。
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return Artifact(name=name, version=directory.name, directory=directory, weights=weights,
                    framework=meta.get("framework", "unknown"), meta=meta)


def list_artifacts(name: str | None = None) -> list[Artifact]:
    """列出已落盘的产物；name 为空时列出全部模型。"""
    if not config.model_dir.is_dir():
        return []
    names = [name] if name else sorted(p.name for p in config.model_dir.iterdir() if p.is_dir())
    out: list[Artifact] = []
    for model_name in names:
        root = _model_root(model_name)
        if not root.is_dir():
            continue
        for directory in sorted([p for p in root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)],
                                key=lambda p: int(p.name[1:])):
            weights = _find_weights(directory)
            if weights is None:
                continue
            meta_path = directory / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            out.append(Artifact(name=model_name, version=directory.name, directory=directory,
                                weights=weights, framework=meta.get("framework", "unknown"), meta=meta))
    return out


def delete_version(name: str, version: str) -> dict:
    """删除某个模型版本目录（危险操作，由 /models/<name>?version= 触发）。

    只允许删 data/models/<模型名>/vN 这种形状，避免误删到别处。
    """
    if not _VERSION_RE.match(version or ""):
        raise ValueError(f"版本号格式不合法：{version!r}（应为 v1、v2 这种）")
    root = _model_root(name).resolve()
    target = (root / version).resolve()
    if target.parent != root:
        raise ValueError("拒绝删除模型目录以外的路径")
    if not target.is_dir():
        raise FileNotFoundError(f"{name} 不存在版本 {version}")
    files = [p for p in target.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    shutil.rmtree(target)
    return {"deleted": f"{name}/{version}", "files": len(files), "freed_kb": round(size / 1024, 1)}
