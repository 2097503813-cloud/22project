# -*- coding: utf-8 -*-
"""数据库写入层：按外键顺序把「训练」和「推理」写进 8 张表。

外键顺序（唯一的建表脚本 sql/schema_mysql.sql 已排好，这里严格照做）：

    ① Datasets ─┐
                ├─▶ ② Models ─▶ ③ Trainings ─┬─▶ ④ ModelInvocations（调用日志）
    ② Models  ──┘                            └─▶ ⑤ InferenceTasks ─▶ ⑥ InferenceResults
                                                   （另需 TargetDatasetID→①、ModelID→②）

关于"「InferenceTasks.TrainingID 与 DeploymentID 谁为权威锚点」"这个老悬案
（它原先记在被删掉的 T-SQL 版 sql/schema.sql 末尾，现把结论落在代码这边），
这里的选择是：**TrainingID 为权威锚点**——推理结果的可信度取决于"哪一次训练"，部署只是
同一次训练的投放位置。因此本服务写库时 DeploymentID / DeviceID 留空（NULL），等「边缘设备」
那条支线真正落地后再回填。

数据库：**只用 MySQL**（pymysql）。早期为"没装库也能跑"写过 SQLite 兜底与 SQL Server(pyodbc)
两条分支，三套方言各自演化出占位符 / LIMIT-TOP / 建表脚本 / 取主键 等一堆差异，维护成本远大于
收益，现已统一：连接参数见 db.env，建表脚本只有 sql/schema_mysql.sql（自带 CREATE DATABASE + USE）。

写库失败一律抛 DBUnavailable / DBError，由上层决定「降级但显式回报」，不允许静默吞掉。
"""

from __future__ import annotations

import json
import re
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal

from .config import config


class DBError(RuntimeError):
    """数据库层通用错误。"""


class DBUnavailable(DBError):
    """连不上或驱动缺失——上层据此降级并回报，而不是静默忽略。"""


def _now() -> str:
    """统一时间戳格式：MySQL 能解析的字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _strip_sql_comments(text: str) -> str:
    """去掉 /* */ 块注释与整行的 -- 行注释，好按分号安全切分脚本。"""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"^\s*--.*$", "", text, flags=re.M)
    return text


def _statements(sql: str) -> list[str]:
    """把建表脚本切成一条条可执行的 SQL（按分号切，切之前先去掉注释，避免注释里的分号捣乱）。

    够用就好的简易切分：schema_mysql.sql 里没有存储过程 / BEGIN…END，不需要真正的 SQL 解析器。
    ⚠️ 它是"裸切"，如果哪天在字符串字面量里写了分号或 `--`，会被切错——建表脚本里保持别这么写。
    """
    return [s.strip() for s in _strip_sql_comments(sql).split(";") if s.strip()]


def _clip(value, limit: int):
    """按列宽截断，避免超长文本直接触发数据库报错。

    规则：只对 str 生效；超长时保留前 `limit - 3` 个字符再补 "..."，让**总长仍 <= limit**，
    正好卡在列宽边界上也不会被 MySQL（严格模式）判成超长。数字 / None / bool 原样透传。
    ⚠️ limit 必须与建表脚本里的列宽一致（如 Description VARCHAR(500)、Status VARCHAR(20)），
    两边对不上就退化成"要么提前截断、要么照样报错"。
    """
    if isinstance(value, str) and len(value) > limit:
        return value[:limit - 3] + "..."
    return value


def _jsonable(value):
    """把驱动返回的对象转成可 JSON 序列化的值。

    pymysql 读 DATETIME 列会返回 datetime 对象、读 DECIMAL 会返回 Decimal、读 BLOB 会返回
    bytes —— 直接塞进 Flask 的 jsonify 会报 "Object of type datetime is not JSON serializable"。
    所有回读接口都必须先过这里（`/trainings` 曾经 500 就是漏了这一步）。
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
    """数据库门面（**只用 MySQL**）：连接按线程复用、幂等建表、写入与回读。

    历史说明：早期支持过 SQLite（零配置兜底）与 SQL Server（pyodbc），
    但三套方言各自演化出一堆分支（占位符 `%s`/`?`、`LIMIT`/`TOP`、建表脚本、
    `PRAGMA foreign_keys`…），维护成本远大于收益，现已统一到 MySQL，
    其他方言在 config 与这里各挡一次。
    """

    def __init__(self, cfg=config) -> None:
        """准备连接缓存与行数缓存；方言不是 MySQL 就直接失败（不猜、不降级）。"""
        self.cfg = cfg
        self.dialect = cfg.db_dialect
        if self.dialect != "mysql":
            raise DBError(f"本项目只支持 MySQL，收到 {self.dialect!r}")
        self._schema_ready = False
        self.last_bootstrap = None      # 记录「顺便建了库/表」的事实，供 /health 展示
        self._local = threading.local()  # 每线程复用一个连接（之前是每个请求都新建连接）
        self._counts_cache = {"at": 0.0, "data": None}

    # ------------------------------------------------------------------ 连接
    @property
    def placeholder(self) -> str:
        """参数占位符：MySQL 用 %s（sqlite/SQL Server 的 `?` 分支已移除）。"""
        return "%s"

    def _ping(self, conn) -> bool:
        """探活：连接超时或数据库重启过时会失败，调用方据此决定重连。"""
        try:
            conn.ping(reconnect=False)
            return True
        except Exception:
            return False

    def _connect(self, require_db: bool = True):
        """取本线程的连接（存在 threading.local 里）；require_db=False 时连到 master（建库前用）。

        复用而不是"每请求新建"：一次 MySQL 握手 + 认证是实打实的开销，早期那种写法是接口变慢的来源之一。
        ⚠️ 缓存必须连 require_db 一起比对：master 连接没执行过 USE，拿它跑业务 SQL 会直接 "No database selected"。
        """
        cached = getattr(self._local, "conn", None)
        # 缓存的连接必须与本次的 require_db 一致：连到 master 的连接不能拿去做业务查询
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
        """建立 MySQL 连接；缺驱动 / 连不上统一抛 DBUnavailable（调用方据此降级）。

        `require_db=False` 用于"库还不存在"的首次启动：先连到服务器，再执行
        `schema_mysql.sql` 里的 CREATE DATABASE + USE。
        """
        cfg = self.cfg
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

    @contextmanager
    def cursor(self, commit: bool = False):
        """借出一个游标；块内正常结束才按 commit 决定提交，异常回滚并把原异常抛出去。

        commit=False 是只读默认值：不提交也无所谓，下次复用同一条连接接着读即可；
        commit=True 时整块当**一个事务**看——要么全落、要么全不落。
        异常必须先 rollback：不退事务的连接会停在"事务开着、InnoDB 行锁没放"的状态，
        一直挂到连接断开，后面同线程的操作会互相等锁。

        finally 里**只关游标、不关连接**：连接是按线程复用的资源，生命周期归 _connect/_ping 管；
        在这里 close 掉，等于把复用的连接又降级回"每请求新建"，白付一次握手认证成本。
        ⚠️ 所以 cursor() 只做语句级隔离，需要跨多次 cursor() 的复合写要自己塞进同一个游标里（见 insert_task_with_results）。
        """
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
        """幂等建表（**只用 MySQL**）：执行 sql/schema_mysql.sql。

        脚本里全部是 `CREATE TABLE IF NOT EXISTS`，可重复执行；库不存在时会自动创建。
        """
        if self._schema_ready:
            return

        # 只用 MySQL：执行 schema_mysql.sql（库不存在就顺手建；脚本可重复执行）
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

        self._schema_ready = True

    def ping(self) -> dict:
        """体检：顺便建表 + 取行数，**永远返回 dict 不抛异常**（/health 靠它保持可用）。"""
        try:
            self.ensure_schema()
            counts = self.table_counts()
            return {"ok": True, "dialect": self.dialect, "counts": counts,
                    "target": f"{self.cfg.db_host}:{self.cfg.db_port}/{self.cfg.db_name}",
                    "bootstrap": self.last_bootstrap}
        except Exception as exc:
            return {"ok": False, "dialect": self.dialect, "error": str(exc),
                    "target": f"{self.cfg.db_host}:{self.cfg.db_port}/{self.cfg.db_name}"}

    # ------------------------------------------------------------------ 写入
    def _insert_returning_id(self, cur, sql: str, params: tuple) -> int:
        """执行 INSERT 并返回自增主键（MySQL 直接用 cursor.lastrowid）。"""
        cur.execute(sql, params)
        return int(cur.lastrowid)

    def _ph(self, n: int) -> str:
        """生成 n 个占位符并用逗号连起来，如 "%s, %s, %s"（VALUES 子句用）。"""
        return ", ".join([self.placeholder] * n)

    def ensure_dataset(self, name: str, source: str | None = None, sample_count: int | None = None,
                       class_count: int | None = None, data_path: str | None = None,
                       description: str | None = None) -> int:
        """① Datasets：有就取、没有就建（幂等），返回 DatasetID 主键。

        按 DatasetName 查（表上有唯一键）：命中直接返回既有主键，否则 INSERT 再回主键。
        做成幂等是因为每次训练/推理都要先把依赖行落实，调用方不必自己判断"该插还是该取"。
        ⚠️ 命中分支**不更新**任何已有字段：同名再传 SampleCount/Description 也是白传，改要走 update_dataset()。
        返回值是主键 ID，给 Trainings.DatasetID / InferenceTasks.TargetDatasetID 当外键用——
        外键顺序必须是 Datasets → Models → Trainings/InferenceTasks，反过来写数据库直接拒绝。
        """
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
        """② Models：有就取、没有就建（幂等），返回 ModelID 主键。

        语义与 ensure_dataset 完全对称：按 ModelName（唯一键）命中就复用，未命中才插。
        ⚠️ 命中分支同样**不更新**已有行：模型描述/类型改了不会因为再调一次而生效。
        传进来的名字必须先过 training.db_model_name() 换算成库里的规范写法（'1DCNN' 而不是 '1dcnn'），
        否则登记与显示的名字会和 schema 种子数据、data/models/<键> 目录名对不上。
        """
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
        """③ Trainings：一次训练写一行，返回 TrainingID（推理任务的外键锚点）。

        字段来源：ModelID/DatasetID 由 ensure_model/ensure_dataset 先落实；Epochs/BatchSize/Accuracy/Loss
        来自训练结果；ModelPath 是 registry 落盘的权重路径；StartedDate/CompletedDate 由调用方给，
        CreatedDate 这里补成写入时刻——失败的那次也照样写一行（Status=失败），便于追溯。
        超长文本一律过 _clip，避免一个长路径/长备注把整次落库带崩。
        """
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
        """④ ModelInvocations：每次 /predict 调用留一条（失败也留），相当于调用审计日志。

        request_params / response_result 是任意结构的 Python 对象，落 LONGTEXT 前先 json.dumps，
        用 default=str 兜住 datetime 之类的非标准类型——审计日志不值得为一个字段 500。
        ⚠️ 布尔列是 TINYINT(1)：is_success 必须显式转 1/0，None 要保留成 NULL，
        不能直接塞 True/False，否则不同驱动下的取值不一致。
        """
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
        """⑤ InferenceTasks 的 SQL 本体（单写、或与结果同事务，两种入口复用）。

        只做 INSERT、不管事务：事务边界由调用方的 cursor() 决定，这样才能既支持"只写任务"，
        也支持 insert_task_with_results 把它和 ⑥ 塞进同一个事务。
        DeploymentID / DeviceID 显式写 None：本服务以 TrainingID 为权威锚点，这两个字段留给边缘设备支线回填。
        """
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
        """⑥ InferenceResults 的 SQL 本体（同上复用），返回写入行数。

        一个任务的明细是"一个窗口一行"，可能几百上千行，所以逐行 execute 最直白；
        也避免拼一条巨型 INSERT 撞上 MySQL 的 max_allowed_packet 而整批失败。
        空 rows 直接返回 0：推理可能一个窗口都没切出来，此时不该报错，也不该写空值行。
        ⚠️ ResultTimestamp 缺省取本次统一的 now，保证同一个任务里所有明细的时间戳一致；
        布尔列同样必须转 1/0；detail 是 JSON 字符串（列宽 500 的 FeatureSnapshot 走 _clip）。
        """
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

    def insert_task_with_results(self, task_kwargs: dict, rows: list[dict]) -> tuple[int, int]:
        """⑤+⑥ **同一个事务**写完任务与全部结果，返回 (task_id, 结果行数)。

        存在的意义：以前是两次独立 commit，结果插入一旦失败，任务行已经按 `Status=成功、Progress=100`
        提交掉了，留下一个"有任务、没结果"的孤儿任务——前端明细点开是空的，日志里还写着成功。
        现在共用一个 cursor（=一个事务），结果失败会连带任务一起 rollback，要么都成功、要么都没写。
        所以推理落库**只走这个入口**：原先还并列着 insert_inference_task / insert_inference_results
        两个"单独写一半"的公开方法，全项目零调用，已删除，免得有人以为落库有第二条路径。
        """
        self.ensure_schema()
        with self.cursor(commit=True) as cur:
            task_id = self._insert_task_cur(cur, **task_kwargs)
            return task_id, self._insert_results_cur(cur, task_id, rows)

    # ------------------------------------------------------------------ 读取
    def _rows_to_dicts(self, cur) -> list[dict]:
        """把游标里剩下的行读成 [{列名: 值}]，每个值都过一遍 _jsonable。

        列名直接取 cursor.description（=SELECT 里的实际列名），所以什么 SQL 都能用这一个函数收口。
        ⚠️ 过 _jsonable 不是可选项：pymysql 会把 DATETIME 读成 datetime、DECIMAL 读成 Decimal、
        BLOB 读成 bytes，原样丢给 Flask 的 jsonify 会抛
        "Object of type datetime is not JSON serializable" —— /trainings 曾经 500 就是漏了这一步。
        """
        cols = [d[0] for d in cur.description]
        return [{c: _jsonable(v) for c, v in zip(cols, row)} for row in cur.fetchall()]

    def latest_training(self, model_name: str | None = None, only_success: bool = True) -> dict | None:
        """取最近一次训练——它同时是推理任务的外键锚点（TrainingID）。

        only_success=True 是默认值，也是这套设计的要害：**失败的训练没有可用权重**，
        拿它当锚点会推理出一个"路径指向不存在的文件"的任务；所以默认只在 Status='成功' 里挑。
        排序按 TrainingID DESC（自增主键即时间序，比按日期列排更稳，不受时钟回拨影响）。
        传 model_name 就只在该模型内挑；两者都没配到时返回 None，由调用方决定报错还是降级。
        """
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
        """按主键取一次训练（显式传 training_id 做推理锚点时会用到）。

        `SELECT *` 是有意的：调用方（推理侧要读 ModelPath/Status/模型类型）需要整行，
        多查几列比再补一个"挑列"的参数简单；这里的行宽很小，不值得为省流量牺牲通用性。
        找不到返回 None，不抛异常——"训练记录不存在"在接口层是要转成 404 的业务事实。
        """
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT * FROM Trainings WHERE TrainingID = {self.placeholder}", (training_id,))
            rows = self._rows_to_dicts(cur)
        return rows[0] if rows else None

    def recent_trainings(self, limit: int = 20) -> list[dict]:
        """最近的训练记录（带模型名、数据集名，供列表页直接显示）。

        LEFT JOIN 是必要的：DatasetID 可空（数据集被删时置空），INNER JOIN 会让这些训练整条消失。
        LIMIT 用 f-string 拼数字而非占位符：参数化值只能出现在"值"的位置，LIMIT 后跟的是字面量，
        好在 limit 已被 int() 强制成整数，拼进去没有注入面。
        """
        self.ensure_schema()
        cols = ("t.TrainingID, t.TrainName, t.Epochs, t.BatchSize, t.Accuracy, t.Loss, t.Status, t.ModelPath, "
                "t.StartedDate, t.CompletedDate, t.CreatedDate, m.ModelName, d.DatasetName")
        # 分页语法两种方言不同：SQL Server 用 TOP，其余用 LIMIT
        tail = "FROM Trainings t LEFT JOIN Models m ON m.ModelID = t.ModelID " \
               "LEFT JOIN Datasets d ON d.DatasetID = t.DatasetID ORDER BY t.TrainingID DESC"
        # 只用 MySQL：直接 LIMIT（原来的 SQL Server `TOP` 分支已移除）
        sql = f"SELECT {cols} {tail} LIMIT {int(limit)}"
        with self.cursor() as cur:
            cur.execute(sql)
            return self._rows_to_dicts(cur)

    def recent_inference_tasks(self, limit: int = 20) -> list[dict]:
        """最近的推理任务（带模型名，供列表页直接显示）。

        与 recent_trainings 同一套路：LEFT JOIN ModelName（上传/占位模型可能不在 Models 里，
        INNER JOIN 会让这些任务从列表里凭空消失），LIMIT 同样经 int() 后再拼进 SQL。
        """
        self.ensure_schema()
        cols = ("k.InferenceTaskID, k.TaskName, k.TaskType, k.Status, k.Progress, k.TrainingID, "
                "k.TargetDatasetID, k.ResultSummary, k.CreatedDate, k.CompletedDate, m.ModelName")
        # 同上：LEFT JOIN 是必要的——上传/占位模型可能在 Models 里查不到
        tail = ("FROM InferenceTasks k LEFT JOIN Models m ON m.ModelID = k.ModelID "
                "ORDER BY k.InferenceTaskID DESC")
        # 只用 MySQL：直接 LIMIT（原来的 SQL Server `TOP` 分支已移除）
        sql = f"SELECT {cols} {tail} LIMIT {int(limit)}"
        with self.cursor() as cur:
            cur.execute(sql)
            return self._rows_to_dicts(cur)

    def inference_task(self, task_id: int) -> dict | None:
        """取一个推理任务及其全部结果明细（明细挂在返回值的 results 里）。

        任务行与明细行分成两条 SELECT，然后手工把 results 挂进 tasks[0]，而不是 JOIN 成一张宽表：
        JOIN 会把任务行按明细行数复制一遍，还得在 Python 里去重。
        ⚠️ 一次把该任务的**全部结果行**返回，没有分页也没有 LIMIT——几千个窗口就是几千行 JSON，
        明细页会明显变慢；要支持大任务得在这里加 offset/limit 并让接口透传。
        任务不存在返回 None（接口层转 404），明细为空就是 results: []，不额外报错。
        """
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT * FROM InferenceTasks WHERE InferenceTaskID = {self.placeholder}", (task_id,))
            tasks = self._rows_to_dicts(cur)
            if not tasks:
                return None
            # 明细可能很多行，按 ResultID 排序保证点开顺序稳定
            cur.execute(f"SELECT * FROM InferenceResults WHERE InferenceTaskID = {self.placeholder} ORDER BY ResultID",
                        (task_id,))
            tasks[0]["results"] = self._rows_to_dicts(cur)
        return tasks[0]

    def models_in_db(self) -> list[dict]:
        """Models 表的全部登记行（供模型清单与「系统管理→数据库」页读）。

        这里**不加 LIMIT**：模型是人工登记的小表（种子数据只有 3 行），清单页必须能看全，
        被截断反而会让人以为模型丢了。行数大的是 Trainings / InferenceResults，不在这个函数里。
        """
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute("SELECT ModelID, ModelName, Description, ApiEndpoint, ModelType, Status, IsActive "
                        "FROM Models ORDER BY ModelID")
            return self._rows_to_dicts(cur)

    def datasets_in_db(self, limit: int = 200) -> list[dict]:
        """Datasets 表登记的数据集（供「数据集管理」页读）。

        ⚠️ 默认 limit=200 是硬上限：第 200 条之后的数据集在这个接口上"查不到"，
        页面看起来像数据丢了（实际在库里）。要翻页得由调用方显式传更大的 limit。
        LIMIT 数字同样经 int() 再拼进 SQL，不占位（LIMIT 后只能跟字面量）。
        """
        self.ensure_schema()
        cols = ("DatasetID, DatasetName, Source, SampleCount, ClassCount, DataPath, Description, CreatedDate")
        tail = "FROM Datasets ORDER BY DatasetID"
        # 只用 MySQL：直接 LIMIT（原来的 SQL Server `TOP` 分支已移除）
        sql = f"SELECT {cols} {tail} LIMIT {int(limit)}"
        with self.cursor() as cur:
            cur.execute(sql)
            return self._rows_to_dicts(cur)

    def register_dataset(self, **kwargs) -> dict:
        """登记数据集，并告知是新建还是已存在（供 POST /datasets/db）。

        先单独 SELECT 一次只为了拿到 already_existed 这个事实：ensure_dataset 本身是幂等的，
        但它只说"ID 是多少"，不区分"这次真建了"还是"本来就有"——前端要靠这个提示用户。
        ⚠️ 探测查询外面套了 try/except 并把 existed 当 False：探测失败时不能连带把登记也搞失败，
        宁可少一个提示，也不能让"登记数据集"这个主操作断在半路。
        """
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

    # ------------------------------------------------------ 模型 CRUD
    # 可被 update_model 改的列白名单——SQL 里 set 的列名只能从这个元组来，
    # 外部传进来的键名一律不当列名用，否则就是一条"任意列名拼进 SQL"的注入面
    _MODEL_FIELDS = ("Description", "ModelType", "ApiEndpoint", "Status", "IsActive")
    # 改名时要一起改的"路径"列（表名 → 列名）：模型改名后磁盘目录跟着改名，
    # 库里这些列存的是旧目录下的绝对/相对路径，不一起换掉，列表页会显示成"产物丢失"
    _PATH_FIELDS = (("Trainings", ("ModelPath",)), ("InferenceTasks", ("InputPath", "OutputPath")),
                    ("ModelDeployments", ("DeployedPath", "DeployUrl")))

    def _count(self, cur, sql: str, params: tuple) -> int:
        """执行 COUNT(*) 取标量（引用统计到处要用，单拎出来）。

        统一 int(...) 转换：COUNT(*) 的返回类型随驱动而异（pymysql 给 int，别家可能给 Decimal），
        转一次就能免掉后面比较/相加时的类型意外。
        fetchone() 为 None 理论上不会发生（COUNT 必有行），兜底成 0 而不是抛异常。
        """
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def model_references(self, name: str) -> dict:
        """某模型被哪些表引用了多少行——删之前必须先看这个，否则外键会直接拒绝。

        逐表 COUNT 而不是一条 UNION：五张表的引用列名不同，分表统计出来的 dict 正好给前端逐项展示。
        ⚠️ 计数口径与 delete_model 的级联口径**不完全一致**：这里按 InferenceResults.ModelID 数，
        而 force 删除走的是"结果 ← 任务"的子查询。如果某条结果行的 ModelID 与它所属任务的 ModelID 不同，
        它既不会被级联删掉、又会在外键上挡住 Models 行的删除（表现为"计数说是 0、删却删不掉"）。
        模型名不存在时抛 DBError 而不是返回空引用——"不存在"和"存在但没人引用"必须能区分。
        """
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
        """Models 表里有没有这个名字（改名查重、上传登记都会用）。

        只取主键判存在，不 SELECT *：调用方只关心 True/False，少读几列少一次反序列化。
        ⚠️ 比较交给 MySQL 做，判重结果受 ModelName 列的 collation 影响：本库是 utf8mb4_unicode_ci，
        **大小写不敏感**，所以 '1dcnn' 与 '1DCNN' 在这里算同名；名字统一走 db_model_name() 更省心。
        """
        self.ensure_schema()
        with self.cursor() as cur:
            cur.execute(f"SELECT ModelID FROM Models WHERE ModelName = {self.placeholder}", (name,))
            return cur.fetchone() is not None

    def rename_model_paths(self, old: str, new: str) -> dict:
        """模型改名后，把库里已存的**路径前缀**一起换掉（...\\models\\old\\... → ...\\models\\new\\...）。

        只动路径列：Trainings.ModelPath、InferenceTasks.InputPath/OutputPath、
        ModelDeployments.DeployedPath/DeployUrl（清单见 _PATH_FIELDS）。
        ⚠️ 为什么必须做：改名时 api._rename_model() 会把磁盘目录一并搬走，库里这些列存的还是旧目录名，
        不换掉就会出现"记录里指着一个已经不存在的路径"，列表页/明细页一致显示产物丢失。
        实现用 SQL 的 REPLACE() 就地改（两种分隔符各来一遍，兼容 Windows 反斜杠与 POSIX 斜杠），
        rowcount 是"被替换到的行数"，全 0 也不报错——本来就可能没有任何记录引用过旧路径。
        ⚠️ touched 的键固定用 cols[0]（该表首列名），而值是所有列累加的行数：
        键名只当"这张表动过"的标记看，不代表只有首列被改。
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
        磁盘上的产物目录要由 api._rename_model() 同步搬，两个动作必须成对调用
        （只改库 = 路径失联，只搬目录 = 名字对不上）。

        列名只能从 _MODEL_FIELDS 白名单来，值一律走占位符——这是防 SQL 注入的关键，别改成拼字符串。
        ⚠️ 值统一按 500 截断，比 Status(20)/ModelType(50) 的列宽宽，超长值这里拦不住，
        会由 MySQL 严格模式报 "Data too long"；换名字时才会先查重，避免直接撞唯一键报错。
        `sets` 为空时主动抛 DBError：静默成功会让调用方以为改生效了（rowcount==0 同理，判"不存在"）。
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
        """删 Models 表登记行。有引用时默认拒绝，force=True 才连带删除引用行。

        默认拒绝而不是默默级联：模型被训练/调用记录引用着，随手删掉会让历史明细全成孤儿。
        force 的删除顺序是**从子到父**（结果 → 任务 → 调用 → 发布 → 训练 → 模型），
        反了就会被外键约束当场拒绝。
        ⚠️ InferenceResults 没有直接的 ModelID 外键约束语义，只能经 InferenceTasks 的子查询间接删；
        子查询与 DELETE 写同一条语句里是有意的——分成两条会在中间留下"任务已删、结果还在"的窗口。
        """
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

    # 说明：原先这里还有 dataset_references / update_dataset / delete_dataset 三个方法
    # （对应 GET/PUT/DELETE /datasets/db/<id>）。三者全项目零调用——前端只在
    # api/platform/index.ts 里声明过 updateDataset/deleteDataset 两个方法，没有任何页面调它们，
    # 路由本身也从来只是"登记的补录入口"，所以整组一并删除。
    # 数据集登记的**写入**仍走 POST /datasets/db，对应下面的 ensure_dataset()。

    def table_counts(self, max_age: float = 30.0) -> dict:
        """8 张表的行数。**带 30 秒缓存**——/health 与 /system 每次都要它，而 8 条 COUNT(*) 在
        MySQL 上不算便宜（之前每个请求都真跑一遍，是页面跳转慢的一个来源）。

        失效方式只有一种：时间到期（`now - at >= max_age`）后下次调用重算，**写入路径不会主动失效**，
        ⚠️ 所以刚写完一次训练，30 秒内的 /health 行数仍可能是旧值——这是有意用"短暂陈旧"换请求延迟，
        要立刻刷新就显式传 max_age=0。
        返回的是 `dict(...)` 浅拷贝：调用方随便改都不会污染缓存（8 个 int 的浅拷贝足够，不需要深拷贝）。
        cache 里连 `at` 一起换掉（整体赋值）而不是分两步改，避免出现"新数据配旧时间戳"的中间态。
        """
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
