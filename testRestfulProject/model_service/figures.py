# -*- coding: utf-8 -*-
"""出图：把训练与推理的结果落成 PNG（无头、不弹窗）。

为什么单独一个模块：
  * 原有把图"表现出来"的地方只有 `plt.show()`（1DCNN.py 的准确率/损失曲线、
    cwt_cnn 的混淆矩阵、adtk 的时序图），进程退出图就没了，也没法在接口里返回；
  * 这里统一用 Agg 后端 + `savefig` 落盘，Flask 跑在无显示环境也能出图，
    图放在 data/figures/<模型>/<版本>/ 下，可通过 GET /figures/<路径> 直接看。

生成内容：
  训练后  training_curves.png（准确率/损失）、confusion_matrix.png（混淆矩阵）、
          per_class_metrics.png（每类 P/R/F1）
  推理后  prediction_distribution.png（预测分布）、predicted_windows.png（窗口波形 + 预测标签）

出图失败一律不影响训练/推理主流程：函数内部捕获异常并返回 error 文本。
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from pathlib import Path

from .config import config

# matplotlib 要在 import pyplot 之前设好后端与缓存目录：
# 缓存目录默认在用户目录下（受限环境可能不可写），统一挪到项目内 data/.cache。
os.environ.setdefault("MPLCONFIGDIR", str(config.data_dir / ".cache" / "matplotlib"))

FIG_DIR = config.data_dir / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_CJK_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"]
_plt = None
_style_ready = False


def _pyplot():
    """懒加载 matplotlib 并配好全局样式（只做一次）。返回 pyplot 模块。"""
    global _plt, _style_ready
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")                      # 无头：不弹窗、不需要显示服务
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 只挂系统里**真的存在**的中文字体：硬写一个不存在的字体名，
        # matplotlib 会静默回落到方框，图上的中文全变成"豆腐块"
        available = {f.name for f in font_manager.fontManager.ttflist}
        cjk = [f for f in _CJK_FONTS if f in available]
        plt.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3
        _plt = plt
        _style_ready = bool(cjk)
    return _plt


def has_cjk() -> bool:
    """本机有没有可用的中文字体（/system 用它提示"图上中文可能显示不全"）。"""
    _pyplot()
    return _style_ready


def _short_labels(labels: list[str]) -> list[str]:
    """把长标签压成适合当坐标轴的两行短名：滚动体故障-0.007in -> 滚动体\\n0.007in"""
    out = []
    for lab in labels:
        text = str(lab)
        if text == "正常":
            out.append("正常")
            continue
        parts = text.split("-")
        kind = parts[0].replace("故障", "")
        detail = parts[1] if len(parts) > 1 else ""
        detail = detail.replace("@6点钟", "@6").replace("in", "")
        out.append(f"{kind}\n{detail}" if detail else kind)
    return out


def _save(fig, path: Path) -> dict:
    """落盘一张图并关掉它（不 close 会累积内存），返回带可直接访问 url 的条目。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    _pyplot().close(fig)
    rel = path.relative_to(FIG_DIR).as_posix()      # 统一成正斜杠，前端当 URL 用
    return {"name": path.name, "path": str(path), "url": f"/figures/{rel}",
            "size_kb": round(path.stat().st_size / 1024, 1)}


# ------------------------------------------------------------------ 训练期出图
def training_figures(model: str, version: str, meta: dict, extra: dict | None = None) -> dict:
    """生成训练期三张图，返回 {"figures": [...], "dir": ..., "error": ...}"""
    extra = extra or {}
    target = FIG_DIR / model / version
    figures: list[dict] = []
    try:
        plt = _pyplot()
        metrics = meta.get("metrics") or {}
        history = metrics.get("history") or {}
        labels = meta.get("labels") or []

        # ① 训练曲线
        epochs = range(1, max((len(v) for v in history.values()), default=0) + 1)
        if epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            pairs = [("accuracy", "val_accuracy", "准确率", axes[0]),
                     ("loss", "val_loss", "损失值", axes[1])]
            for key, val_key, title, ax in pairs:
                if key in history:
                    ax.plot(epochs, history[key], "-.", color="#c0392b", label=f"训练集{title}")
                if val_key in history:
                    ax.plot(epochs, history[val_key], "b-.", label=f"验证集{title}")
                ax.set_title(f"训练集与验证集{title}曲线")
                ax.set_xlabel("训练轮次")
                ax.set_ylabel(title)
                ax.legend()
            fig.suptitle(f"{meta.get('model', model)} {version} · 训练曲线 "
                         f"(epochs={len(list(epochs))})", fontsize=13)
            figures.append(_save(fig, target / "training_curves.png"))

        # ② 混淆矩阵
        confusion = extra.get("confusion")
        if confusion and labels:
            import numpy as np
            mat = np.asarray(confusion, dtype=int)
            short = _short_labels(labels)
            fig, ax = plt.subplots(figsize=(8.6, 7.2))
            im = ax.imshow(mat, cmap="Blues")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(short)), short, fontsize=8)
            ax.set_xlabel("预测类别")
            ax.set_ylabel("真实类别")
            ax.set_title(f"{meta.get('model', model)} {version} · 混淆矩阵"
                         f"（测试集 {int(mat.sum())} 个窗口，准确率 "
                         f"{metrics.get('test_accuracy')}）", fontsize=12)
            threshold = mat.max() / 2 if mat.size else 0
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if mat[i, j]:
                        ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > threshold else "#333333")
            ax.grid(False)
            figures.append(_save(fig, target / "confusion_matrix.png"))

        # ③ 每类 P/R/F1
        per_class = extra.get("per_class") or {}
        if per_class and labels:
            short = _short_labels(labels)
            cols = [c for c in ("precision", "recall", "f1-score") if any(
                c in per_class.get(str(i), {}) for i in range(len(labels)))]
            if cols:
                import numpy as np
                x = np.arange(len(labels))
                width = 0.8 / len(cols)
                fig, ax = plt.subplots(figsize=(11, 4.6))
                for k, col in enumerate(cols):
                    values = [float(per_class.get(str(i), {}).get(col, 0) or 0) for i in range(len(labels))]
                    ax.bar(x + k * width, values, width, label=col)
                ax.set_xticks(x + width * (len(cols) - 1) / 2, short, rotation=45, ha="right", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("得分")
                ax.set_title(f"{meta.get('model', model)} {version} · 各类别精确率/召回率/F1")
                ax.legend()
                figures.append(_save(fig, target / "per_class_metrics.png"))

        return {"figures": figures, "dir": str(target), "error": None}
    except Exception as exc:                                     # pragma: no cover
        return {"figures": figures, "dir": str(target),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-800:]}


# ------------------------------------------------------------------ 推理期出图
def inference_figures(model: str, version: str, payload: dict, matrix=None) -> dict:
    """生成推理期两张图：预测分布 + 预测窗口波形（最多 6 个窗口）。"""
    target = FIG_DIR / model / version / f"predict-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    figures: list[dict] = []
    try:
        plt = _pyplot()
        predictions = payload.get("predictions") or []
        task = payload.get("task")
        summary = payload.get("summary") or {}

        # ① 预测分布
        if task == "classification":
            from collections import Counter
            counter = Counter(p["predicted_label"] for p in predictions)
        else:
            counter = Counter(p["predicted_label"] for p in predictions)
        if counter:
            keys = list(counter)
            values = [counter[k] for k in keys]
            colors = ["#27ae60" if k in ("正常",) else "#c0392b" for k in keys]
            fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(keys) + 3), 4.2))
            bars = ax.bar(range(len(keys)), values, color=colors, width=0.6)
            ax.bar_label(bars)
            ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("窗口数")
            ax.set_title(f"{model} {version} · 本次推理 {payload.get('count')} 个窗口的预测分布")
            figures.append(_save(fig, target / "prediction_distribution.png"))

        # ② 预测窗口波形
        if matrix is not None:
            import numpy as np
            matrix = np.asarray(matrix)
            show = min(6, len(matrix))
            cols = 2
            rows = (show + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(12, 2.0 * rows), squeeze=False)
            for i in range(show):
                ax = axes[i // cols][i % cols]
                pred = predictions[i]
                signal = matrix[i]
                ax.plot(range(len(signal)), signal, lw=0.7, color="#2c3e50")
                if task == "classification":
                    title = f"窗口{pred['index']} → {pred['predicted_label']} ({pred['confidence']:.3f})"
                    if pred.get("actual_class") is not None:
                        hit = pred["predicted_class"] == pred["actual_class"]
                        # 注意：✓/✗ 这类符号在雅黑/宋体里没有字形，会渲染成方框，所以用中文标注
                        title += "  命中" if hit else "  未命中"
                        ax.set_facecolor("#eafaf1" if hit else "#fdedec")
                else:
                    title = (f"窗口{pred['index']} → {pred['predicted_label']}"
                             f"（异常点占比 {pred['anomaly_score']:.3f}）")
                    ax.set_facecolor("#fdedec" if pred["is_anomaly"] else "#eafaf1")
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("采样点", fontsize=8)
                ax.set_ylabel("幅值", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2)
            for j in range(show, rows * cols):
                axes[j // cols][j % cols].axis("off")
            fig.suptitle(f"{model} {version} · 预测窗口原始信号（绿底=命中/正常，红底=未命中/异常）"
                         f"\n{str(payload.get('input', {}).get('path') or '内联数组')}", fontsize=11)
            fig.tight_layout()
            figures.append(_save(fig, target / "predicted_windows.png"))

        return {"figures": figures, "dir": str(target), "error": None}
    except Exception as exc:                                     # pragma: no cover
        return {"figures": figures, "dir": str(target),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-800:]}


def list_figures(limit: int = 200) -> list[dict]:
    """列出已生成的图，供 GET /figures 用。"""
    out = []
    for path in sorted(FIG_DIR.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = path.relative_to(FIG_DIR).as_posix()
        out.append({"file": rel, "url": f"/figures/{rel}",
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
        if len(out) >= limit:
            break
    return out


def clear_figures() -> dict:
    """清空图库（危险操作，由系统的「维护」触发）。目录本身保留。"""
    removed = 0
    freed = 0
    for path in list(FIG_DIR.rglob("*.png")):
        freed += path.stat().st_size
        path.unlink(missing_ok=True)
        removed += 1
    for directory in sorted([p for p in FIG_DIR.rglob("*") if p.is_dir()],
                            key=lambda p: len(p.parts), reverse=True):
        try:
            directory.rmdir()                       # 只删空目录
        except OSError:
            pass
    return {"removed": removed, "freed_kb": round(freed / 1024, 1), "dir": str(FIG_DIR)}
