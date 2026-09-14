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
    """\u4e00\u4e2a\u5df2\u843d\u76d8\u7684\u6a21\u578b\u4ea7\u7269\uff1a\u5b9a\u4f4d\u4fe1\u606f\uff08\u76ee\u5f55/\u6743\u91cd\u6587\u4ef6\uff09+ \u81ea\u89e3\u91ca\u4fe1\u606f\uff08\u6846\u67b6/meta\uff09\u3002"""

    name: str
    version: str
    directory: Path
    weights: Path
    framework: str
    meta: dict = field(default_factory=dict)

    @property
    def meta_path(self) -> Path:
        """该产物的 meta.json 路径（可能还不存在，save_artifact 时才写）。"""
        return self.directory / "meta.json"

    def to_dict(self) -> dict:
        """\u6311\u51fa\u7ed9\u63a5\u53e3/\u524d\u7aef\u7528\u7684\u5b57\u6bb5\uff08meta \u91cc\u7684\u539f\u59cb dict \u592a\u5927\uff0c\u4e0d\u900f\u4f20\uff09\u3002"""
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
    """取下一个可用版本目录，形如 data/models/1dcnn/v2。"""
    root = _model_root(name)
    root.mkdir(parents=True, exist_ok=True)
    used = [int(m.group(1)) for p in root.iterdir() if p.is_dir() and (m := _VERSION_RE.match(p.name))]
    version = f"v{max(used, default=0) + 1}"
    target = root / version
    target.mkdir(parents=True, exist_ok=False)
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


def save_artifact(name: str, framework: str, saver, meta: dict, keep_previous: bool = True) -> Artifact:
    """落盘一个模型产物。

    saver: 可调用对象，签名 saver(target_dir: Path) -> Path（返回权重文件路径）。
    失败时会把刚建的版本目录清掉，不留半个产物。
    """
    target = next_version_dir(name)
    try:
        weights = saver(target)
        meta = {
            **meta,
            "model": name,
            "version": target.name,
            "framework": framework,
            "weights_file": Path(weights).name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        artifact = Artifact(name=name, version=target.name, directory=target,
                            weights=Path(weights), framework=framework, meta=meta)
        if not keep_previous:
            _prune_except(name, keep=target.name)
        return artifact
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _prune_except(name: str, keep: str) -> None:
    """只保留 keep 这一个版本，其余版本目录整个删掉（keep_previous=False 时用）。"""
    for p in _model_root(name).iterdir():
        if p.is_dir() and p.name != keep:
            shutil.rmtree(p, ignore_errors=True)


def load_artifact(name: str, version: str | None = None) -> Artifact:
    """按名字（可指定版本）取产物。version 为空时取最新。"""
    root = _model_root(name)
    if not root.is_dir():
        raise FileNotFoundError(f"还没有任何 {name} 的模型产物，先调 /train")
    if version:
        directory = root / version
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} 不存在版本 {version}")
    else:
        candidates = sorted([p for p in root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)],
                            key=lambda p: int(p.name[1:]))
        if not candidates:
            raise FileNotFoundError(f"还没有任何 {name} 的模型产物，先调 /train")
        directory = candidates[-1]

    weights = _find_weights(directory)
    if weights is None:
        raise FileNotFoundError(f"{directory} 里找不到权重文件")
    meta_path = directory / "meta.json"
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


def latest_meta(name: str) -> dict:
    """最新版本的 meta（拿不到产物时抛 FileNotFoundError，由调用方决定怎么报）。"""
    return load_artifact(name).meta


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
