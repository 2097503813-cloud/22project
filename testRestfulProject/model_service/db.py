# -*- coding: utf-8 -*-
"""数据库写入层：按外键顺序把「训练」和「推理」写进 8 张表。

外键顺序（sql/schema.sql 与 schema_mysql.sql 已排好，这里严格照做）：

    ① Datasets ─┐
                ├─▶ ② Models ─▶ ③ Trainings ─┬─▶ ④ ModelInvocations（调用日志）
    ② Models  ──┘                            └─▶ ⑤ InferenceTasks ─▶ ⑥ InferenceResults
                                                   （另需 TargetDatasetID→①、ModelID→②）

关于 schema.sql 末尾自己标注的悬案「InferenceTasks.TrainingID 与 DeploymentID 谁为权威锚点」，
这里的选择是：**TrainingID 为权威锚点**——推理结果的可信度取决于"哪一次训练"，部署只是
同一次训练的投放位置。因此本服务写库时 DeploymentID / DeviceID 留空（NULL），等「边缘设备」
那条支线真正落地后再回填。

三种方言：
    sqlite     开发兜底，零配置，用 sql/schema_sqlite.sql 建镜像表
    mysql      走 pymysql，执行 sql/schema_mysql.sql（脚本自带 CREATE DATABASE + USE）
    sqlserver  走 pyodbc，执行 sql/schema.sql（按 GO 分批；库不存在则先在 master 里建）

写库失败一律抛 DBUnavailable / DBError，由上层决定「降级但显式回报」，不允许静默吞掉。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from .config import config


class DBError(RuntimeError):
    """数据库层通用错误。"""


class DBUnavailable(DBError):
    """连不上或驱动缺失——上层据此降级并回报，而不是静默忽略。"""


def _now() -> str:
    """统一时间戳格式：MySQL / SQL Server / SQLite 都能解析的字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _strip_sql_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"^\s*--.*$", "", text, flags=re.M)
    return text


def _statements(sql: str) -> list[str]:
    return [s.strip() for s in _strip_sql_comments(sql).split(";") if s.strip()]


def _clip(value, limit: int):
    """按列宽截断，避免超长文本直接触发数据库报错。"""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit - 3] + "..."
    return value


def _jsonable(value):
    """把驱动返回的对象转成可 JSON 序列化的值。

    SQLite 返回的是字符串，而 MySQL(pymysql) 会返回 datetime、SQL Server(pyodbc)
    还可能返回 Decimal/bytes——回读接口必须统一，否则 Flask 的 json 编码器会直接报
    "Object of type datetime is not JSON serializable"。
    """
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date,)):
        return value.isoformat()
    return value


class Database:
    def __init__(self, cfg=config) -> None:
        self.cfg = cfg
        self.dialect = cfg.db_dialect
        if self.dialect not in ("sqlite", "mysql", "sqlserver"):
            raise DBError(f"不支持的 MODEL_DB_DIALECT：{self.dialect}")
        self._schema_ready = False
        self.last_bootstrap = None      # 记录「顺便建了库/表」的事实，供 /health 展示
        self._local = threading.local()  # 每线程复用一个连接（之前是每个请求都新建连接）
        self._counts_cache = {"at": 0.0, "data": None}

    # ------------------------------------------------------------------ 连接
    @property
    def placeholder(self) -> str:
        return "%s" if self.dialect == "mysql" else "?"

    def _ping(self, conn) -> bool:
        try:
            if self.dialect == "mysql":
                conn.ping(reconnect=False)
            elif self.dialect == "sqlite":
                conn.execute("SELECT 1")
            else:
                conn.cursor().execute("SELECT 1")
            return True
        except Exception:
            return False

    def _connect(self, require_db: bool = True):
        cfg = self.cfg
        cached = getattr(self._local, "conn", None)
        if cached is not None and getattr(self._local, "req_db", None) == require_db:
            if self._ping(cached):
                return cached
            try:                                   # 连接失效（超时/重启）就先丢弃再重连
                cached.close()
            except Exception:
                pass
            self._local.conn = None
        conn = self._connect_new(require_db)
        self._local.conn = conn
        self._local.req_db = require_db
        return conn

    def _connect_new(self, require_db: bool = True):
        cfg = self.cfg
        if self.dialect == "sqlite":
            conn = sqlite3.connect(str(cfg.sqlite_path), timeout=15)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")     # 让外键顺序真的被校验
            return conn

        if self.dialect == "mysql":
            try:
                import pymysql
            except ImportError as exc:
                raise DBUnavailable("缺少 pymysql 驱动：pip install pymysql") from exc
            kwargs = dict(host=cfg.db_host, port=cfg.db_port, user=cfg.db_user,
                          password=cfg.db_password, charset="utf8mb4", autocommit=False)
            if require_db:
                kwargs["database"] = cfg.db_name
            try:
                return pymysql.connect(**kwargs)
            except Exception as exc:
                raise DBUnavailable(
                    f"MySQL 连接失败({cfg.db_host}:{cfg.db_port}/{cfg.db_name if require_db else '-'})：{exc}") from exc

        try:
            import pyodbc
        except ImportError as exc:
            raise DBUnavailable("缺少 pyodbc 驱动：pip install pyodbc") from exc
        auth = ("Trusted_Connection=yes;" if cfg.db_trusted else f"UID={cfg.db_user};PWD={cfg.db_password};")
        db = cfg.db_name if require_db else "master"
        cs = (f"DRIVER={{{cfg.db_odbc_driver}}};SERVER={cfg.db_host},{cfg.db_port};DATABASE={db};"
              f"{auth}Encrypt=no;TrustServerCertificate=yes;")
        try:
            return pyodbc.connect(cs, timeout=8)
        except Exception as exc:
            raise DBUnavailable(f"SQL Server 连接失败({db})：{exc}") from exc

    @contextmanager
    def cursor(self, commit: bool = False):
        """借出一个游标。注意：**不再每次关闭连接**——连接按线程复用，由 _connect 统一管理。"""
        conn = self._connect()
        cur = conn.cursor()
        try:
            yield cur
            if commit:
                conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

    # -------------------------------------------------------------- 建表/体检
    def ensure_schema(self) -> None:
        """幂等建表：三个方言各跑各自的脚本（脚本本身都可重复执行）。"""
        if self._schema_ready:
            return

        if self.dialect == "sqlite":
            sql = (self.cfg.sql_dir / "schema_sqlite.sql").read_text(encoding="utf-8")
            with self.cursor(commit=True) as cur:
                cur.executescript(sql)
            self.last_bootstrap = f"sqlite 镜像表就绪：{self.cfg.sqlite_path}"

        elif self.dialect == "mysql":
            sql = (self.cfg.sql_dir / "schema_mysql.sql").read_text(encoding="utf-8")
            try:
                conn = self._connect()                      # 库已存在
            except DBUnavailable:
                conn = self._connect(require_db=False)      # 库还不存在：脚本里有 CREATE DATABASE + USE
                self.last_bootstrap = f"MySQL 库 {self.cfg.db_name} 由 schema_mysql.sql 顺手创建"
            try:
                cur = conn.cursor()
                for stmt in _statements(sql):
                    cur.execute(stmt)
                conn.commit()
            finally:
                conn.close()

        else:
            sql = (self.cfg.sql_dir / "schema.sql").read_text(encoding="utf-8")
            batches = [b.strip() for b in re.split(r"^\s*GO\s*$", _strip_sql_comments(sql), flags=re.M | re.I) if b.strip()]
            try:
                conn = self._connect()
            except DBUnavailable:
                boot = self._connect(require_db=False)
                try:
                    boot.autocommit = True
                    boot.cursor().execute(
                        f"IF DB_ID(N'{self.cfg.db_name}') IS NULL CREATE DATABASE [{self.cfg.db_name}]")
                    self.last_bootstrap = f"SQL Server 库 {self.cfg.db_name} 由服务顺手创建"
                finally:
                    boot.close()
                conn = self._connect()
            try:
                cur = conn.cursor()
                for batch in batches:
                    cur.execute(batch)
                conn.commit()
            finally:
                conn.close()

        self._schema_ready = True

    def ping(self) -> dict:
        try:
            self.ensure_schema()
            counts = self.table_counts()
            return {"ok": True, "dialect": self.dialect, "counts": counts,
                    "target": str(self.cfg.sqlite_path) if self.dialect == "sqlite"
                    else f"{self.cfg.db_host}:{self.cfg.db_port}/{self.cfg.db_name}",
                    "bootstrap": self.last_bootstrap}
        except Exception as exc:
            return {"ok": False, "dialect": self.dialect, "error": str(exc),
                    "target": str(self.cfg.sqlite_path) if self.dialect == "sqlite"
                    else f"{self.cfg.db_host}:{self.cfg.db_port}/{self.cfg.db_name}"}

    # ------------------------------------------------------------------ 写入
    def _insert_returning_id(self, cur, sql: str, params: tuple) -> int:
        cur.execute(sql, params)
        if self.dialect == "sqlserver":
            cur.execute("SELECT CAST(SCOPE_IDENTITY() AS INT)")
            return int(cur.fetchone()[0])
        return int(cur.lastrowid)

    def _ph(self, n: int) -> str:
        return ", ".join([self.placeholder] * n)

    def ensure_dataset(self, name: str, source: str | None = None, sample_count: int | None = None,
                       class_count: int | None = None, data_path: str | None = None,
                       description: str | None = None) -> int:
        """① Datasets：有就取，没有就建。"""
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            cur.execute(f"SELECT DatasetID FROM Datasets WHERE DatasetName = {self.placeholder}", (name,))
            row = cur.fetchone()
            if row:
                return int(row[0])
            return self._insert_returning_id(
                cur,
                "INSERT INTO Datasets (DatasetName, Source, SampleCount, ClassCount, DataPath, Description, CreatedDate) "
                f"VALUES ({self._ph(7)})",
                (name, _clip(source, 200), sample_count, class_count, _clip(data_path, 500),
                 _clip(description, 500), _now()),
            )

    def ensure_model(self, name: str, description: str | None = None, model_type: str | None = None,
                     api_endpoint: str | None = "/predict", status: str | None = "可运行") -> int:
        """② Models：有就取，没有就建。"""
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            cur.execute(f"SELECT ModelID FROM Models WHERE ModelName = {self.placeholder}", (name,))
            row = cur.fetchone()
            if row:
                return int(row[0])
            return self._insert_returning_id(
                cur,
                "INSERT INTO Models (ModelName, Description, ApiEndpoint, ModelType, CreatedDate, IsActive, Status) "
                f"VALUES ({self._ph(7)})",
                (name, _clip(description, 500), _clip(api_endpoint, 255), _clip(model_type, 50), _now(), 1,
                 _clip(status, 20)),
            )

    def insert_training(self, model_id: int, dataset_id: int | None, train_name: str,
                        epochs: int | None, batch_size: int | None, accuracy: float | None,
                        loss: float | None, model_path: str | None, status: str,
                        started: str | None = None, completed: str | None = None,
                        created_by: str | None = "model_service", remark: str | None = None) -> int:
        """③ Trainings。"""
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            return self._insert_returning_id(
                cur,
                "INSERT INTO Trainings (ModelID, DatasetID, TrainName, Epochs, BatchSize, Accuracy, Loss, "
                "ModelPath, Status, CreatedDate, StartedDate, CompletedDate, CreatedBy, Remark) "
                f"VALUES ({self._ph(14)})",
                (model_id, dataset_id, _clip(train_name, 200), epochs, batch_size, accuracy, loss,
                 _clip(model_path, 500), _clip(status, 20), _now(), started, completed,
                 _clip(created_by, 100), _clip(remark, 500)),
            )

    def insert_invocation(self, model_id: int, training_id: int | None, api_endpoint: str,
                          request_params, response_result=None, duration_ms: int | None = None,
                          is_success: bool | None = None, status_code: int | None = None,
                          error_message: str | None = None, client_ip: str | None = None,
                          status: str | None = None) -> int:
        """④ ModelInvocations：每次 /predict 调用留一条。"""
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            return self._insert_returning_id(
                cur,
                "INSERT INTO ModelInvocations (ModelID, TrainingID, ApiEndpoint, RequestParams, ResponseResult, "
                "DurationMs, IsSuccess, StatusCode, ErrorMessage, ClientIP, Status, InvocationDate) "
                f"VALUES ({self._ph(12)})",
                (model_id, training_id, _clip(api_endpoint, 255),
                 json.dumps(request_params, ensure_ascii=False, default=str),
                 json.dumps(response_result, ensure_ascii=False, default=str) if response_result is not None else None,
                 duration_ms, None if is_success is None else (1 if is_success else 0), status_code,
                 error_message, _clip(client_ip, 50), _clip(status, 20), _now()),
            )

    def _insert_task_cur(self, cur, training_id: int, target_dataset_id: int, task_name: str,
                         task_type: str, status: str, inference_params=None, result_summary=None,
                         error_message: str | None = None, model_id: int | None = None,
                         input_path: str | None = None, output_path: str | None = None,
                         progress: int | None = None, started: str | None = None,
                         completed: str | None = None, created_by: str | None = "model_service") -> int:
        """⑤ InferenceTasks 的 SQL 本体（单写、或与结果同事务，两种入口复用）。"""
        return self._insert_returning_id(
            cur,
            "INSERT INTO InferenceTasks (TrainingID, TargetDatasetID, TaskName, TaskType, Status, "
            "InferenceParams, ResultSummary, ErrorMessage, DeploymentID, ModelID, DeviceID, InputPath, "
            "OutputPath, Progress, CreatedDate, StartedDate, CompletedDate, CreatedBy) "
            f"VALUES ({self._ph(18)})",
            (training_id, target_dataset_id, _clip(task_name, 200), _clip(task_type, 20), _clip(status, 20),
             json.dumps(inference_params, ensure_ascii=False, default=str) if inference_params is not None else None,
             json.dumps(result_summary, ensure_ascii=False, default=str) if result_summary is not None else None,
             error_message, None, model_id, None, _clip(input_path, 500), _clip(output_path, 500), progress,
             _now(), started, completed, _clip(created_by, 100)),
        )

    def _insert_results_cur(self, cur, task_id: int, rows: list[dict]) -> int:
        """⑥ InferenceResults 的 SQL 本体（同上复用）。"""
        if not rows:
            return 0
        cols = ("InferenceTaskID", "RowIdentifier", "ResultTimestamp", "PredictedValue", "AnomalyScore",
                "IsAnomaly", "PredictedCategory", "Confidence", "FeatureSnapshot", "ModelID", "SampleIndex",
                "PredictedClass", "PredictedLabel", "Score", "ActualClass", "ResultDetail", "CreatedDate")
        sql = (f"INSERT INTO InferenceResults ({', '.join(cols)}) VALUES ({self._ph(len(cols))})")
        now = _now()
        for row in rows:
            cur.execute(sql, (
                task_id, _clip(row.get("row_identifier"), 100), row.get("result_timestamp") or now,
                row.get("predicted_value"), row.get("anomaly_score"),
                None if row.get("is_anomaly") is None else (1 if row["is_anomaly"] else 0),
                _clip(row.get("predicted_category"), 50), row.get("confidence"),
                _clip(row.get("feature_snapshot"), 500), row.get("model_id"), row.get("sample_index"),
                row.get("predicted_class"), _clip(row.get("predicted_label"), 100), row.get("score"),
                row.get("actual_class"),
                json.dumps(row.get("detail"), ensure_ascii=False, default=str) if row.get("detail") is not None else None,
                now,
            ))
        return len(rows)

    def insert_inference_task(self, **kwargs) -> int:
        """⑤ 单独写一条 InferenceTasks（推理落库请改用 insert_task_with_results，保证同事务）。"""
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            return self._insert_task_cur(cur, **kwargs)

    def insert_inference_results(self, task_id: int, rows: list[dict]) -> int:
        """⑥ 单独写 InferenceResults 明细。"""
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            return self._insert_results_cur(cur, task_id, rows)

    def insert_task_with_results(self, task_kwargs: dict, rows: list[dict]) -> tuple[int, int]:
        """⑤+⑥ **一个事务**写完任务与全部结果。

        以前是两次独立 commit：结果插入失败时，任务行已经提交成 `Status=成功、Progress=100`，
        留下一个"有任务、没结果"的孤儿任务，前端明细点开是空的。
        """
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            task_id = self._insert_task_cur(cur, **task_kwargs)
            return task_id, self._insert_results_cur(cur, task_id, rows)

    # ------------------------------------------------------------------ 读取
    def _rows_to_dicts(self, cur) -> list[dict]:
        cols = [d[0] for d in cur.description]
        return [{c: _jsonable(v) for c, v in zip(cols, row)} for row in cur.fetchall()]

    def latest_training(self, model_name: str | None = None, only_success: bool = True) -> dict | None:
        """取最近一次训练——它同时是推理任务的外键锚点（TrainingID）。"""
        self.ensure_schema()
        sql = ("SELECT t.TrainingID, t.ModelID, t.DatasetID, t.TrainName, t.Accuracy, t.Loss, t.ModelPath, "
               "t.Status, t.CreatedDate, m.ModelName "
               "FROM Trainings t JOIN Models m ON m.ModelID = t.ModelID WHERE 1 = 1")
        params: list = []
        if model_name:
            sql += f" AND m.ModelName = {self.placeholder}"
            params.append(model_name)
        if only_success:
            sql += f" AND t.Status = {self.placeholder}"
            params.append("成功")
        sql += " ORDER BY t.TrainingID DESC"
        with self.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = self._rows_to_dicts(cur)
        return rows[0] if rows else None

    def training_by_id(self, training_id: int) -> dict | None:
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT * FROM Trainings WHERE TrainingID = {self.placeholder}", (training_id,))
            rows = self._rows_to_dicts(cur)
        return rows[0] if rows else None

    def recent_trainings(self, limit: int = 20) -> list[dict]:
        self.ensure_schema()
        cols = ("t.TrainingID, t.TrainName, t.Epochs, t.BatchSize, t.Accuracy, t.Loss, t.Status, t.ModelPath, "
                "t.StartedDate, t.CompletedDate, t.CreatedDate, m.ModelName, d.DatasetName")
        tail = "FROM Trainings t LEFT JOIN Models m ON m.ModelID = t.ModelID " \
               "LEFT JOIN Datasets d ON d.DatasetID = t.DatasetID ORDER BY t.TrainingID DESC"
        sql = (f"SELECT TOP {int(limit)} {cols} {tail}" if self.dialect == "sqlserver"
               else f"SELECT {cols} {tail} LIMIT {int(limit)}")
        with self.cursor() as cur:
            cur.execute(sql)
            return self._rows_to_dicts(cur)

    def recent_inference_tasks(self, limit: int = 20) -> list[dict]:
        self.ensure_schema()
        cols = ("k.InferenceTaskID, k.TaskName, k.TaskType, k.Status, k.Progress, k.TrainingID, "
                "k.TargetDatasetID, k.ResultSummary, k.CreatedDate, k.CompletedDate, m.ModelName")
        tail = ("FROM InferenceTasks k LEFT JOIN Models m ON m.ModelID = k.ModelID "
                "ORDER BY k.InferenceTaskID DESC")
        sql = (f"SELECT TOP {int(limit)} {cols} {tail}" if self.dialect == "sqlserver"
               else f"SELECT {cols} {tail} LIMIT {int(limit)}")
        with self.cursor() as cur:
            cur.execute(sql)
            return self._rows_to_dicts(cur)

    def inference_task(self, task_id: int) -> dict | None:
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT * FROM InferenceTasks WHERE InferenceTaskID = {self.placeholder}", (task_id,))
            tasks = self._rows_to_dicts(cur)
            if not tasks:
                return None
            cur.execute(f"SELECT * FROM InferenceResults WHERE InferenceTaskID = {self.placeholder} ORDER BY ResultID",
                        (task_id,))
            tasks[0]["results"] = self._rows_to_dicts(cur)
        return tasks[0]

    def models_in_db(self) -> list[dict]:
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute("SELECT ModelID, ModelName, Description, ApiEndpoint, ModelType, Status, IsActive "
                        "FROM Models ORDER BY ModelID")
            return self._rows_to_dicts(cur)

    def datasets_in_db(self, limit: int = 200) -> list[dict]:
        """Datasets 表登记的数据集（供「数据集管理」页读）。"""
        self.ensure_schema()
        cols = ("DatasetID, DatasetName, Source, SampleCount, ClassCount, DataPath, Description, CreatedDate")
        tail = "FROM Datasets ORDER BY DatasetID"
        sql = (f"SELECT TOP {int(limit)} {cols} {tail}" if self.dialect == "sqlserver"
               else f"SELECT {cols} {tail} LIMIT {int(limit)}")
        with self.cursor() as cur:
            cur.execute(sql)
            return self._rows_to_dicts(cur)

    def register_dataset(self, **kwargs) -> dict:
        """登记数据集，并告知是新建还是已存在（供 POST /datasets/db）。"""
        name = kwargs.get("name")
        existed = False
        self.ensure_schema()
        try:
            with self.cursor() as cur:
                cur.execute(f"SELECT DatasetID FROM Datasets WHERE DatasetName = {self.placeholder}", (name,))
                existed = cur.fetchone() is not None
        except Exception:
            existed = False
        dataset_id = self.ensure_dataset(
            name=name, source=kwargs.get("source"), sample_count=kwargs.get("sample_count"),
            class_count=kwargs.get("class_count"), data_path=kwargs.get("data_path"),
            description=kwargs.get("description"))
        return {"DatasetID": dataset_id, "already_existed": existed, "DatasetName": name}

    # ------------------------------------------------------ 模型/数据集 CRUD
    _MODEL_FIELDS = ("Description", "ModelType", "ApiEndpoint", "Status", "IsActive")
    _DATASET_FIELDS = ("Source", "SampleCount", "ClassCount", "DataPath", "Description")
    # 改名时要一起改的"路径"列（表名 → 列名）
    _PATH_FIELDS = (("Trainings", ("ModelPath",)), ("InferenceTasks", ("InputPath", "OutputPath")),
                    ("ModelDeployments", ("DeployedPath", "DeployUrl")))

    def _count(self, cur, sql: str, params: tuple) -> int:
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def model_references(self, name: str) -> dict:
        """某模型被哪些表引用了多少行——删之前必须先看这个，否则外键会直接拒绝。"""
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT ModelID FROM Models WHERE ModelName = {self.placeholder}", (name,))
            row = cur.fetchone()
            if not row:
                raise DBError(f"模型 {name} 不存在于 Models 表")
            mid = int(row[0])
            refs = {}
            for table, col in (("Trainings", "ModelID"), ("ModelInvocations", "ModelID"),
                               ("ModelDeployments", "ModelID"), ("InferenceTasks", "ModelID"),
                               ("InferenceResults", "ModelID")):
                refs[table] = self._count(cur, f"SELECT COUNT(*) FROM {table} WHERE {col} = {self.placeholder}", (mid,))
            return {"ModelID": mid, "references": refs,
                    "total": sum(refs.values()), "deletable": sum(refs.values()) == 0}

    def model_exists(self, name: str) -> bool:
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT ModelID FROM Models WHERE ModelName = {self.placeholder}", (name,))
            return cur.fetchone() is not None

    def rename_model_paths(self, old: str, new: str) -> dict:
        """模型改名后，把库里已存的**路径前缀**一起换掉（...\\models\\old\\... → ...\\models\\new\\...）。

        只动路径列：Trainings.ModelPath、InferenceTasks.InputPath/OutputPath、
        ModelDeployments.DeployedPath/DeployUrl。用 SQL 的 REPLACE() 直接改，
        返回值是各表实际被改动的行数。
        """
        self.ensure_schema()
        pairs = ((f"\\models\\{old}\\", f"\\models\\{new}\\"), (f"/models/{old}/", f"/models/{new}/"))
        touched: dict[str, int] = {}
        with self.cursor(commit=True) as cur:
            for table, cols in self._PATH_FIELDS:
                hits = 0
                for col in cols:
                    for old_pat, new_pat in pairs:
                        cur.execute(f"UPDATE {table} SET {col} = REPLACE({col}, {self.placeholder}, {self.placeholder})",
                                    (old_pat, new_pat))
                        hits += max(cur.rowcount, 0)
                if hits:
                    touched[f"{table}.{cols[0]}"] = hits
        return touched

    def update_model(self, name: str, fields: dict, new_name: str | None = None) -> dict:
        """改 Models 表的登记信息（**不碰产物文件**）。

        改名：传 `new_name`，这里只负责「查重 + 改 ModelName」；
        磁盘上的产物目录要由 api._rename_model() 同步搬，两个动作必须成对调用。
        """
        self.ensure_schema()
        sets, params = [], []
        for col in self._MODEL_FIELDS:
            if col in fields and fields[col] is not None:
                sets.append(f"{col} = {self.placeholder}")
                params.append(_clip(fields[col], 500))
        if new_name and new_name != name:
            if self.model_exists(new_name):
                raise DBError(f"模型名 {new_name} 已被占用，换一个")
            sets.append(f"ModelName = {self.placeholder}")
            params.append(_clip(new_name, 100))
        if not sets:
            raise DBError("没有可更新的字段（支持：" + ", ".join(self._MODEL_FIELDS)
                          + "，改名传 NewModelName）")
        params.append(name)
        with self.cursor(commit=True) as cur:
            cur.execute(f"UPDATE Models SET {', '.join(sets)} WHERE ModelName = {self.placeholder}", tuple(params))
            if cur.rowcount == 0:
                raise DBError(f"模型 {name} 不存在于 Models 表")
        return {"updated": name, "new_name": new_name or name, "fields": [s.split(" =")[0] for s in sets]}

    def delete_model(self, name: str, force: bool = False) -> dict:
        """删 Models 表登记行。有引用时默认拒绝，force=True 才连带删除引用行。"""
        self.ensure_schema()
        refs = self.model_references(name)
        if refs["total"] and not force:
            raise DBError(f"模型 {name} 仍被引用（{refs['references']}），"
                          f"要么先清理这些记录，要么用 force=true 连带删除")
        mid = refs["ModelID"]
        with self.cursor(commit=True) as cur:
            if force:
                for table in ("InferenceResults", "InferenceTasks", "ModelInvocations",
                              "ModelDeployments", "Trainings"):
                    # InferenceResults 需要经由 InferenceTasks 间接关联
                    if table == "InferenceResults":
                        cur.execute(
                            "DELETE FROM InferenceResults WHERE InferenceTaskID IN "
                            f"(SELECT InferenceTaskID FROM InferenceTasks WHERE ModelID = {self.placeholder})", (mid,))
                    else:
                        cur.execute(f"DELETE FROM {table} WHERE ModelID = {self.placeholder}", (mid,))
            cur.execute(f"DELETE FROM Models WHERE ModelID = {self.placeholder}", (mid,))
        return {"deleted": name, "cascaded": bool(force and refs["total"]), "references": refs["references"]}

    def dataset_references(self, dataset_id: int) -> dict:
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT DatasetID, DatasetName FROM Datasets WHERE DatasetID = {self.placeholder}", (dataset_id,))
            row = cur.fetchone()
            if not row:
                raise DBError(f"DatasetID={dataset_id} 不存在")
            refs = {
                "Trainings": self._count(cur, f"SELECT COUNT(*) FROM Trainings WHERE DatasetID = {self.placeholder}", (dataset_id,)),
                "InferenceTasks": self._count(cur, f"SELECT COUNT(*) FROM InferenceTasks WHERE TargetDatasetID = {self.placeholder}", (dataset_id,)),
            }
            return {"DatasetID": int(row[0]), "DatasetName": row[1], "references": refs,
                    "total": sum(refs.values()), "deletable": sum(refs.values()) == 0}

    def update_dataset(self, dataset_id: int, fields: dict) -> dict:
        self.ensure_schema()
        sets, params = [], []
        for col in self._DATASET_FIELDS:
            if col in fields and fields[col] is not None:
                sets.append(f"{col} = {self.placeholder}")
                params.append(_clip(fields[col], 500))
        if "DatasetName" in fields and fields["DatasetName"]:
            sets.append(f"DatasetName = {self.placeholder}")
            params.append(_clip(fields["DatasetName"], 100))
        if not sets:
            raise DBError("没有可更新的字段")
        params.append(dataset_id)
        with self.cursor(commit=True) as cur:
            cur.execute(f"UPDATE Datasets SET {', '.join(sets)} WHERE DatasetID = {self.placeholder}", tuple(params))
            if cur.rowcount == 0:
                raise DBError(f"DatasetID={dataset_id} 更新失败（不存在或值未变化）")
        return {"updated": dataset_id, "fields": [s.split(" =")[0] for s in sets]}

    def delete_dataset(self, dataset_id: int, force: bool = False) -> dict:
        self.ensure_schema()
        refs = self.dataset_references(dataset_id)
        if refs["total"] and not force:
            raise DBError(f"数据集 {refs['DatasetName']} 仍被引用（{refs['references']}），"
                          f"先清理或改用 force=true")
        with self.cursor(commit=True) as cur:
            if force:
                cur.execute(f"UPDATE Trainings SET DatasetID = NULL WHERE DatasetID = {self.placeholder}", (dataset_id,))
                cur.execute("DELETE FROM InferenceResults WHERE InferenceTaskID IN "
                            f"(SELECT InferenceTaskID FROM InferenceTasks WHERE TargetDatasetID = {self.placeholder})",
                            (dataset_id,))
                cur.execute(f"DELETE FROM InferenceTasks WHERE TargetDatasetID = {self.placeholder}", (dataset_id,))
            cur.execute(f"DELETE FROM Datasets WHERE DatasetID = {self.placeholder}", (dataset_id,))
        return {"deleted": refs["DatasetName"], "DatasetID": dataset_id,
                "cascaded": bool(force and refs["total"])}

    def table_counts(self, max_age: float = 30.0) -> dict:
        """8 张表的行数。**带 30 秒缓存**——/health 与 /system 每次都要它，而 8 条 COUNT(*) 在
        MySQL 上不算便宜（之前每个请求都真跑一遍，是页面跳转慢的一个来源）。"""
        now = time.time()
        if self._counts_cache["data"] is not None and now - self._counts_cache["at"] < max_age:
            return dict(self._counts_cache["data"])
        self.ensure_schema()
        out: dict[str, int] = {}
        with self.cursor() as cur:
            for table in ("Datasets", "Models", "Trainings", "ModelInvocations",
                          "ModelDeployments", "InferenceTasks", "InferenceResults", "EdgeDevices"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                out[table] = int(cur.fetchone()[0])
        self._counts_cache = {"at": now, "data": out}
        return dict(out)


database = Database()
