# -*- coding: utf-8 -*-
"""临时自测：模型探测（是不是模型）+ 上传（免填 input_len/类别数）+ 改名。

跑法：venv\\Scripts\\python.exe _selftest_upload.py
跑完全部自己清理（删产物目录 + 删 Models 登记行），不留垃圾。
"""
from __future__ import annotations

import io
import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from flask import Flask                                             # noqa: E402
from flask_restful import Api                                       # noqa: E402

from model_service.api import probe_weight, register_api            # noqa: E402
from model_service.config import config as CONF                     # noqa: E402
from model_service.db import database                               # noqa: E402

TEST_NAMES = ("SELFTEST-H5", "SELFTEST-PT", "SELFTEST-BAD", "SELFTEST-TXT", "SELFTEST-H5-RENAMED")

OK, FAIL = "✅", "❌"
failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"  {OK if cond else FAIL} {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        failures.append(name)


def fake_h5(input_len: int = 512, units: int = 10) -> bytes:
    """造一个 Keras HDF5：有 model_weights 组 + model_config 属性（含输入形状与 Dense units）。"""
    import h5py
    config = {"class_name": "Sequential", "config": {"layers": [
        {"class_name": "InputLayer", "config": {"batch_input_shape": [None, input_len, 1]}},
        {"class_name": "Conv1D", "config": {"filters": 8, "kernel_size": 3}},
        {"class_name": "Flatten", "config": {}},
        {"class_name": "Dense", "config": {"units": units}},
    ]}}
    buf = io.BytesIO()
    with h5py.File(buf, "w") as handle:
        handle.create_group("model_weights")
        handle.attrs["model_config"] = json.dumps(config)
    return buf.getvalue()


def fake_pt(units: int = 3) -> bytes:
    import torch
    from torch import nn
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, units))
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.getvalue()


def main() -> int:
    print("=== 0) 清理上一轮残留 ===")
    for name in TEST_NAMES:
        directory = CONF.model_dir / name
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            print(f"  删残留目录 {directory}")
        try:
            database.delete_model(name, force=True)
            print(f"  删残留登记 {name}")
        except Exception:                                            # noqa: BLE001
            pass

    print("\n=== 1) probe_weight：判断'这是不是一个模型' ===")
    cases = [
        ("model.h5", fake_h5(512, 10), True, {"input_len": 512, "num_classes": 10}),
        ("model.pt", fake_pt(3), True, {"num_classes": 3}),
        ("detector.pkl", pickle.dumps({"k": 1}), True, {}),
        ("notes.txt", b"hello world", False, {}),                     # 后缀不认
        ("fake.pt", b"this is definitely not a torch checkpoint", False, {}),
        ("random.h5", b"\x89HDF\r\n\x1a\n not really hdf5 body", False, {}),
        ("empty.h5", b"", False, {}),
    ]
    for filename, blob, expect_ok, expect in cases:
        info = probe_weight(filename, blob)
        check(f"{filename} → ok={info['ok']}", info["ok"] is expect_ok, info["reason"][:90])
        for key, value in expect.items():
            check(f"  {filename} 读出 {key}={info[key]}", info[key] == value)

    print("\n=== 2) POST /models/upload：不填 input_len 也能落盘 ===")
    app = Flask(__name__)
    register_api(Api(app))
    client = app.test_client()
    created: list[str] = []

    def upload(name: str, files: list[tuple[str, bytes]]):
        # 用 MultiDict 让多个文件共用同一个字段名 file（浏览器多选/选文件夹就是这么发的）
        from werkzeug.datastructures import MultiDict
        payload = MultiDict()
        payload.add("name", name)
        payload.add("description", "自测用，稍后删除")
        for filename, blob in files:
            payload.add("file", (io.BytesIO(blob), filename))
        return client.post("/models/upload", data=payload, content_type="multipart/form-data")

    res = upload("SELFTEST-H5", [("model.h5", fake_h5(512, 10)), ("scaler.npz", b"PK\x03\x04fake")])
    body = res.get_json()
    check("H5 上传返回 201", res.status_code == 201, str(body)[:160])
    if res.status_code == 201:
        created.append(body["model"])
        check("input_len 自动识别 = 512", body["input_len"] == 512, f"actual={body['input_len']}")
        check("类别数自动识别 = 10", body["num_classes"] == 10, f"actual={body['num_classes']}")
        check("10 类自动补 CWRU 中文标签", len(body["labels"] or []) == 10, str(body["labels"])[:80])
        check("落盘 meta.json 存在", (Path(body["directory"]) / "meta.json").is_file())
        check("Models 表已登记", any(r["ModelName"] == body["model"] for r in database.models_in_db()))
        check("ModelType 规范化成 Classification", body["db"]["written"] and
              next(r["ModelType"] for r in database.models_in_db() if r["ModelName"] == body["model"]) == "Classification")

    res = upload("SELFTEST-PT", [("model.pt", fake_pt(3))])
    body = res.get_json()
    check("PT 上传返回 201", res.status_code == 201, str(body)[:160])
    if res.status_code == 201:
        created.append(body["model"])
        check("PT 类别数从 fc.weight 读出 = 3", body["num_classes"] == 3, f"actual={body['num_classes']}")
        check("PT 读不出 input_len 时给 warning", bool(body["warnings"]), str(body["warnings"])[:120])

    res = upload("SELFTEST-BAD", [("model.pt", b"\x00\x01\x02 not a model")])
    body = res.get_json()
    check("假 .pt 被拒（400）", res.status_code == 400, str(body.get("error"))[:80])
    check("拒绝时附带每个文件的失败原因", bool(body.get("detail")), json.dumps(body.get("detail"), ensure_ascii=False)[:150])

    res = upload("SELFTEST-TXT", [("readme.txt", b"just text")])
    check("只传 txt → 400「没有权重文件」", res.status_code == 400, str(res.get_json().get("error"))[:80])

    print("\n=== 3) PUT /models/<名>：改模型名（连产物目录一起搬）===")
    if created:
        old = created[0]
        old_dir = CONF.model_dir / old
        check("改名前进产目录存在", old_dir.is_dir(), str(old_dir))
        res = client.put(f"/models/{old}", json={"NewModelName": "SELFTEST-H5-RENAMED", "Description": "改名自测"})
        body = res.get_json()
        check("改名返回 200", res.status_code == 200, str(body)[:200])
        rename = (body or {}).get("rename") or {}
        check("产物目录已搬迁", rename.get("artifact_dir_moved") is True, str(rename))
        check("老目录不存在了", not old_dir.exists())
        new_dir = CONF.model_dir / "SELFTEST-H5-RENAMED"
        check("新目录里有 meta.json", (new_dir / "v1" / "meta.json").is_file(), str(new_dir))
        meta = json.loads((new_dir / "v1" / "meta.json").read_text(encoding="utf-8"))
        check("meta.json 的 model 字段已改", meta.get("model") == "SELFTEST-H5-RENAMED", str(meta.get("model")))
        names = [r["ModelName"] for r in database.models_in_db()]
        check("Models 表里旧名字没了、新名字在", old not in names and "SELFTEST-H5-RENAMED" in names)
        body = client.put("/models/SELFTEST-H5-RENAMED", json={"NewModelName": "1DCNN"}).get_json()
        check("改成已占用的名字被拒（409）", body.get("error") is not None and "占用" in str(body.get("error")), str(body)[:100])
        res = client.put("/models/SELFTEST-H5-RENAMED", json={"Description": "只改描述"})
        check("只改描述仍然可用（不传 NewModelName）", res.status_code == 200, str(res.get_json())[:120])
        created.append("SELFTEST-H5-RENAMED")

    print("\n=== 4) 清理 ===")
    for name in created:
        try:
            body = client.delete(f"/models/{name}?scope=record&force=true").get_json()
            print(f"  删登记行 {name} → {body}")
        except Exception as exc:                                     # noqa: BLE001
            print(f"  删登记行 {name} 失败：{exc}")
        directory = CONF.model_dir / name
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            print(f"  删产物目录 {directory}")

    print("\n" + ("全部通过 ✅" if not failures else f"有 {len(failures)} 项失败 ❌：{failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
