# -*- coding: utf-8 -*-
"""运行期配置。

所有可变项都走环境变量，代码里只留「本机开发默认值」：

    MODEL_DB_DIALECT    sqlite | mysql | sqlserver        （默认 sqlite）
    MODEL_DB_HOST       默认 127.0.0.1
    MODEL_DB_PORT       默认 3306（mysql）/ 1433（sqlserver）
    MODEL_DB_USER       默认 root
    MODEL_DB_PASSWORD   默认空
    MODEL_DB_NAME       默认 model_management
    MODEL_DB_ODBC_DRIVER  SQL Server 用，默认 "ODBC Driver 18 for SQL Server"
    MODEL_DB_TRUSTED     SQL Server 用 Windows 信任连接时设 1

默认走 sqlite 是刻意的：项目里没有任何数据库连接代码、也没有依赖清单，
先让「训练→推理」这条链在零配置下可验证；要接真库只改环境变量即可，
表结构直接复用 sql/schema_mysql.sql、sql/schema.sql。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVICE_DIR.parent            # testRestfulProject
WORKSPACE_DIR = PROJECT_DIR.parent          # 22project

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = DATA_DIR / "models"             # 模型产物（对应图里的「Pxl模型」）
LOG_DIR = DATA_DIR / "logs"                 # 训练日志
UPLOAD_DIR = DATA_DIR / "datasets"          # 上传/存放的表格数据集（一个子目录 = 一个数据集）
SQL_DIR = PROJECT_DIR / "sql"
SQLITE_PATH = DATA_DIR / "model_management.db"

# 内置的 CWRU .mat 数据集：键是前端/接口里用的数据集名，值是磁盘目录
DATASET_DIRS = {
    "CWRU-0HP": PROJECT_DIR / "1DCNN" / "0HP",
    "CWRU-0HP(cwt)": PROJECT_DIR / "cwt_cnn" / "0HP",
}
ADTK_DATASET_DIR = PROJECT_DIR / "adtk" / "dataset"

# 这三个目录是运行期必需品，import 时就建好，免得别处还要各自判存在性
for _d in (DATA_DIR, MODEL_DIR, LOG_DIR, UPLOAD_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_env_file() -> str | None:
    """可选的本地配置：testRestfulProject/db.env。

    格式为 `KEY=VALUE`（# 开头为注释），只在不与已有环境变量冲突时生效，
    这样开发时不用每次都 export 一串 MODEL_DB_*。已存在同名环境变量则以环境变量为准。
    """
    path = PROJECT_DIR / "db.env"
    if not path.is_file():
        return None
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return str(path) if loaded else None


ENV_FILE = load_env_file()


def ensure_writable_tempdir() -> str:
    """确认临时目录可写，不可写就整体退到项目内的 data/tmp。

    起因：Keras `model.save()` 会先写一个 NamedTemporaryFile 再改名，在受限环境
    （沙箱 / 严格 ACL）下这一步会抛 PermissionError，导致"训练成功但模型存不下来"。
    这里只在探测失败时才切换，正常机器上行为不变。
    """
    def _writable(path: str) -> bool:
        """在 path 里真写一个临时文件试试——mkdir 成功不等于能写（ACL 可能只给读）。"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, delete=True) as fh:
                fh.write(b"probe")
            return True
        except Exception:
            return False

    current = tempfile.gettempdir()
    if _writable(current):
        return current
    fallback = DATA_DIR / "tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(fallback)
    tempfile.tempdir = str(fallback)
    return str(fallback)


TEMP_DIR = ensure_writable_tempdir()


def _env(name: str, default: str | None = None) -> str | None:
    """读环境变量；空字符串按"没设"处理（否则 MODEL_DB_PASSWORD= 会被当成真的是空密码）。"""
    v = os.getenv(name)
    return v if v not in (None, "") else default


class Config:
    """一次进程生命周期内不变的服务配置。"""

    def __init__(self) -> None:
        """把模块级的路径常量与环境变量快照成一份不可变配置。"""
        self.service_dir = SERVICE_DIR
        self.project_dir = PROJECT_DIR
        self.workspace_dir = WORKSPACE_DIR
        self.data_dir = DATA_DIR
        self.model_dir = MODEL_DIR
        self.log_dir = LOG_DIR
        self.upload_dir = UPLOAD_DIR
        self.sql_dir = SQL_DIR
        self.dataset_dirs = dict(DATASET_DIRS)
        self.adtk_dataset_dir = ADTK_DATASET_DIR

        # 本项目**只支持 MySQL**：早期为了"没装库也能跑"写过 SQLite 兜底与 SQL Server 分支，
        # 结果是三套方言各自演化、埋了不少坑（占位符、TOP/LIMIT、建表语句）。现在统一到 MySQL，
        # 别的取值直接报错，免得有人配错了却"看起来能跑"。
        self.db_dialect = (_env("MODEL_DB_DIALECT", "mysql") or "mysql").lower()
        if self.db_dialect != "mysql":
            raise RuntimeError(f"本项目只支持 MySQL（MODEL_DB_DIALECT=mysql），收到 {self.db_dialect!r}；"
                               f"请检查 testRestfulProject/db.env")
        # 连接参数（MySQL 默认端口 3306；账号密码放 db.env，不入库）
        self.db_host = _env("MODEL_DB_HOST", "127.0.0.1")
        self.db_user = _env("MODEL_DB_USER", "root")
        self.db_password = _env("MODEL_DB_PASSWORD", "")
        self.db_name = _env("MODEL_DB_NAME", "model_management")
        self.db_port = int(_env("MODEL_DB_PORT", "3306") or "3306")

        # 训练/推理默认超参，分别沿用两个脚本原有的取值，保证与既有实验可比
        self.defaults = {
            "1dcnn": {
                "dataset": "CWRU-0HP", "length": 784, "number": 600, "stride": 150,
                "rate": [0.7, 0.15, 0.15], "normal": True, "epochs": 10, "batch_size": 128,
            },
            "cwt_cnn": {
                "dataset": "CWRU-0HP", "length": 784, "number": 300, "stride": 150,
                "rate": [0.5, 0.25, 0.25], "normal": True, "epochs": 50, "batch_size": 0,
            },
            "adtk": {
                # 无监督：把「正常」信号切成窗口，窗口当样本，fit adtk 的 PcaAD
                # （k=4 主成分；feature_mode=stats 用 10 维统计特征，raw 则直接用 784 点原始幅值）
                "dataset": "CWRU-0HP", "length": 784, "number": 600, "stride": 784,
                "detector": "PcaAD", "k": 4, "c": 5.0, "feature_mode": "stats",
                "sampling_rate": 48000, "threshold_quantile": 0.995, "factor": 1.0,
                "baseline_file": "normal_0_97.mat",
            },
        }

    # ---- 便于 /health 与日志展示 ----
    def describe(self) -> dict:
        """给 /health 与前端「运行信息」用的配置摘要。

        数据库连接信息（只用 MySQL；账号密码放 db.env，不入库）。
        """
        return {
            "project_dir": str(self.project_dir),
            "model_dir": str(self.model_dir),
            "temp_dir": TEMP_DIR,
            "env_file": ENV_FILE,
            "db": {
                "dialect": self.db_dialect,
                "host": self.db_host if self.db_dialect != "sqlite" else None,
                "port": self.db_port if self.db_dialect != "sqlite" else None,
                "database": self.db_name if self.db_dialect != "sqlite" else str(self.sqlite_path),
            },
            "datasets": {k: str(v) for k, v in self.dataset_dirs.items()},
            "upload_dir": str(self.upload_dir),
        }


config = Config()
