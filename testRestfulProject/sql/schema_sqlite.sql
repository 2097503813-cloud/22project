/* ================================================================
   模型管理库 · SQLite 版（开发/离线兜底用，与 schema_mysql.sql 同构）
   ----------------------------------------------------------------
   用途：在没接真库（MySQL / SQL Server）时，让 model_service 能把
        「训练 → 推理」的结果真实落库，从而端到端验证外键顺序。
   说明：
     - 表名、列名、外键命名与 schema_mysql.sql 完全一致，方便日后切换
     - SQLite 无 DATETIME 类型，统一用 TEXT 存 ISO8601 字符串
     - SQLite 无 AUTO_INCREMENT 语义差异，用 INTEGER PRIMARY KEY AUTOINCREMENT
     - 外键约束需 db.py 里 PRAGMA foreign_keys=ON 才生效
   ================================================================ */

/* ---------------- 1. Datasets ---------------- */
/* 一个数据集一行。DatasetName 唯一，db.ensure_dataset() 就是按它做"有就取、没有就建" */
CREATE TABLE IF NOT EXISTS `Datasets` (
    `DatasetID`    INTEGER PRIMARY KEY AUTOINCREMENT,
    `DatasetName`  TEXT NOT NULL UNIQUE,      -- 数据集名，训练/推理都用它做外键锚点
    `Source`       TEXT,                      -- 来源（CWRU 官方数据集 / 用户上传 / ADHOC 临时）
    `SampleCount`  INTEGER,                   -- 样本（窗口）数，体检结果里回填
    `ClassCount`   INTEGER,                   -- 类别数；无监督模型为 NULL
    `DataPath`     TEXT,                      -- 数据目录（落库时已脱敏成工作区内相对路径）
    `Description`  TEXT,
    `CreatedDate`  TEXT DEFAULT (datetime('now','localtime'))
);

/* ---------------- 2. Models ---------------- */
/* 模型登记表。ModelName 唯一；上传/训练时按名字"有就取用、不重复插" */
CREATE TABLE IF NOT EXISTS `Models` (
    `ModelID`     INTEGER PRIMARY KEY AUTOINCREMENT,
    `ModelName`   TEXT NOT NULL UNIQUE,
    `Description` TEXT,
    `ApiEndpoint` TEXT,                       -- 该模型的调用入口，一般为 /predict
    `ModelType`   TEXT,                       -- Classification / AnomalyDetection / Regression
    `CreatedDate` TEXT DEFAULT (datetime('now','localtime')),
    `IsActive`    INTEGER DEFAULT 1,
    `Status`      TEXT                        -- 可运行 / 未训练 …
);

/* ---------------- 3. EdgeDevices ---------------- */
/* 边缘设备。当前服务不写这张表（「边缘设备」支线尚未落地），保留结构供后续部署使用 */
CREATE TABLE IF NOT EXISTS `EdgeDevices` (
    `DeviceID`      INTEGER PRIMARY KEY AUTOINCREMENT,
    `DeviceName`    TEXT NOT NULL,
    `DeviceType`    TEXT,                     -- 设备类型（工控机 / 网关 …）
    `IPAddress`     TEXT,                     -- IPv4/IPv6 文本，故用 TEXT 而非定长
    `Location`      TEXT,                     -- 部署位置
    `HardwareSpecs` TEXT,                     -- 硬件规格（自由文本/JSON）
    `EdgeStatus`    TEXT,                     -- 设备侧的运行状态（与下面的 Status 语义重叠，见 schema.sql 备注）
    `LastHeartbeat` TEXT,                     -- 最后一次心跳时间
    `DeviceCode`    TEXT,                     -- 设备编号
    `MacAddress`    TEXT,
    `OsVersion`     TEXT,
    `Status`        TEXT,                     -- 本平台的登记状态
    `IsActive`      INTEGER DEFAULT 1,
    `Remark`        TEXT,
    `CreatedDate`   TEXT DEFAULT (datetime('now','localtime'))
);

/* ---------------- 4. Trainings ---------------- */
/* 训练记录。被 ModelInvocations / ModelDeployments / InferenceTasks 三张表引用，
   是整套结构的中枢：InferenceTasks.TrainingID 就是这里的"权威锚点" */
CREATE TABLE IF NOT EXISTS `Trainings` (
    `TrainingID`    INTEGER PRIMARY KEY AUTOINCREMENT,
    `ModelID`       INTEGER NOT NULL REFERENCES `Models`(`ModelID`),      -- 外键，必须已登记
    `DatasetID`     INTEGER REFERENCES `Datasets`(`DatasetID`),           -- 可空：删除数据集时会置空
    `TrainName`     TEXT,                     -- 形如 1dcnn-20250912-193000
    `Epochs`        INTEGER,
    `BatchSize`     INTEGER,
    `Accuracy`      REAL,                     -- 测试集准确率；无监督模型为 NULL
    `Loss`          REAL,
    `ModelPath`     TEXT,                     -- 权重文件路径，指向 data/models/<模型>/vN/…
    `Status`        TEXT,                     -- 成功 / 失败
    `CreatedDate`   TEXT DEFAULT (datetime('now','localtime')),
    `StartedDate`   TEXT,
    `CompletedDate` TEXT,
    `CreatedBy`     TEXT,
    `Remark`        TEXT                      -- 跳过越界窗口数、NaN 窗口数、日志/图目录等 JSON
);

/* ---------------- 5. ModelInvocations ---------------- */
/* 每次 /predict 留一条（失败也留）。可以看作最简单的调用审计日志 */
CREATE TABLE IF NOT EXISTS `ModelInvocations` (
    `InvocationID`   INTEGER PRIMARY KEY AUTOINCREMENT,
    `ModelID`        INTEGER NOT NULL REFERENCES `Models`(`ModelID`),
    `TrainingID`     INTEGER REFERENCES `Trainings`(`TrainingID`),   -- 能定位到哪次训练时才有值
    `ApiEndpoint`    TEXT,                    -- 目前都是 /predict
    `RequestParams`  TEXT NOT NULL,           -- 请求参数 JSON（samples 体积太大，不记）
    `ResponseResult` TEXT,                    -- 成功时的摘要 / 失败时的错误 JSON
    `DurationMs`     INTEGER,                 -- 本次耗时（毫秒）；失败路径可能为 NULL
    `IsSuccess`      INTEGER,                 -- 1/0
    `StatusCode`     INTEGER,                 -- HTTP 状态码
    `ErrorMessage`   TEXT,
    `ClientIP`       TEXT,
    `Status`         TEXT,                    -- 成功 / 失败
    `InvocationDate` TEXT DEFAULT (datetime('now','localtime'))
);

/* ---------------- 6. ModelDeployments ---------------- */
/* 模型发布记录（模型投放到某台边缘设备）。当前服务同样不写这张表 */
CREATE TABLE IF NOT EXISTS `ModelDeployments` (
    `DeploymentID`    INTEGER PRIMARY KEY AUTOINCREMENT,
    `ModelID`         INTEGER NOT NULL REFERENCES `Models`(`ModelID`),
    `TrainingID`      INTEGER NOT NULL REFERENCES `Trainings`(`TrainingID`),   -- 发布的是哪一次训练的产物
    `DeviceID`        INTEGER NOT NULL REFERENCES `EdgeDevices`(`DeviceID`),
    `Version`         TEXT,                   -- 版本号，如 v2
    `VersionAlias`    TEXT,                   -- 别名，如 latest / stable
    `Environment`     TEXT,                   -- 环境：prod / test …
    `DeployUrl`       TEXT,                   -- 部署后的访问地址
    `DeployedPath`    TEXT,                   -- 设备上的产物路径
    `ServicePort`     INTEGER CHECK (`ServicePort` IS NULL OR (`ServicePort` BETWEEN 1 AND 65535)),
    `RuntimeParams`   TEXT,                   -- 运行参数（JSON）
    `IsActive`        INTEGER DEFAULT 1,
    `IsCurrent`       INTEGER NOT NULL DEFAULT 1,   -- 是否该模型的当前生效版本
    `DeployStatus`    TEXT,
    `LastStatusCheck` TEXT,
    `DeployedDate`    TEXT DEFAULT (datetime('now','localtime')),
    `DeployedBy`      TEXT,
    `ErrorMessage`    TEXT,
    `Remark`          TEXT
);

/* ---------------- 7. InferenceTasks ---------------- */
/* 一次推理 = 一个任务，明细在 InferenceResults。
   TrainingID / TargetDatasetID 都是 NOT NULL 外键，所以推理前必须已有成功的训练记录 */
CREATE TABLE IF NOT EXISTS `InferenceTasks` (
    `InferenceTaskID`  INTEGER PRIMARY KEY AUTOINCREMENT,
    `TrainingID`       INTEGER NOT NULL REFERENCES `Trainings`(`TrainingID`),     -- 权威锚点
    `TargetDatasetID`  INTEGER NOT NULL REFERENCES `Datasets`(`DatasetID`),       -- 内联样本会登记成 ADHOC-<模型> 数据集
    `TaskName`         TEXT NOT NULL,         -- 形如 predict-1dcnn-20250912-193000
    `TaskType`         TEXT NOT NULL,         -- classification / anomaly_detection（取自产物 meta）
    `Status`           TEXT,
    `InferenceParams`  TEXT,                  -- 请求参数 JSON
    `ResultSummary`    TEXT,                  -- 样本数、预测分布、模型版本…
    `ErrorMessage`     TEXT,
    `DeploymentID`     INTEGER REFERENCES `ModelDeployments`(`DeploymentID`),     -- 当前留空，等边缘设备支线落地后回填
    `ModelID`          INTEGER REFERENCES `Models`(`ModelID`),
    `DeviceID`         INTEGER REFERENCES `EdgeDevices`(`DeviceID`),              -- 同上，留空
    `InputPath`        TEXT,                  -- 输入来源文件路径
    `OutputPath`       TEXT,                  -- 出图目录
    `Progress`         INTEGER CHECK (`Progress` IS NULL OR (`Progress` BETWEEN 0 AND 100)),
    `CreatedDate`      TEXT DEFAULT (datetime('now','localtime')),
    `StartedDate`      TEXT,
    `CompletedDate`    TEXT,
    `CreatedBy`        TEXT
);

/* ---------------- 8. InferenceResults ---------------- */
/* 推理结果明细，一个任务 N 行（一个窗口一行）。
   分类任务用 PredictedClass/Label/Confidence，异常检测任务用 IsAnomaly/AnomalyScore，
   两类共享同一张表，所以近义列较多（见 schema.sql 末尾的冗余列备注） */
CREATE TABLE IF NOT EXISTS `InferenceResults` (
    `ResultID`          INTEGER PRIMARY KEY AUTOINCREMENT,
    `InferenceTaskID`   INTEGER NOT NULL REFERENCES `InferenceTasks`(`InferenceTaskID`),
    `RowIdentifier`     TEXT,                 -- 形如 内圈故障.csv#3，用来回溯这一行是哪来的
    `ResultTimestamp`   TEXT,                 -- 该行结果的时间（取写入时刻）
    `PredictedValue`    REAL,                 -- 分类任务里的置信度（同类名字段，见备注）
    `AnomalyScore`      REAL,                 -- 异常检测：窗口重构误差 / 异常点占比
    `IsAnomaly`         INTEGER,              -- 1/0
    `PredictedCategory` TEXT,                 -- Classification / AnomalyDetection
    `Confidence`        REAL,
    `FeatureSnapshot`   TEXT,                 -- top_k 的 JSON 快照（超 500 字符则只记 truncated）
    `ModelID`           INTEGER REFERENCES `Models`(`ModelID`),
    `SampleIndex`       INTEGER,              -- 窗口序号
    `PredictedClass`    INTEGER,              -- 类别号
    `PredictedLabel`    TEXT,                 -- 类别中文名
    `Score`             REAL,                 -- 与 Confidence 同值
    `ActualClass`       INTEGER,              -- 输入来自登记过的数据集文件时才有真值
    `ResultDetail`      TEXT,                 -- 判定细节 JSON（阈值、重构误差、相对倍数…）
    `CreatedDate`       TEXT DEFAULT (datetime('now','localtime'))
);

/* ---------------- 索引（与 MySQL 版同名） ---------------- */
CREATE INDEX IF NOT EXISTS `IX_Trainings_ModelID`                 ON `Trainings`(`ModelID`);
CREATE INDEX IF NOT EXISTS `IX_ModelInvocations_ModelID_Date`     ON `ModelInvocations`(`ModelID`, `InvocationDate`);
CREATE INDEX IF NOT EXISTS `IX_ModelDeployments_ModelID_Current`  ON `ModelDeployments`(`ModelID`, `IsCurrent`);
CREATE INDEX IF NOT EXISTS `IX_ModelDeployments_DeviceID`         ON `ModelDeployments`(`DeviceID`);
CREATE INDEX IF NOT EXISTS `IX_InferenceTasks_Status`             ON `InferenceTasks`(`Status`);
CREATE INDEX IF NOT EXISTS `IX_InferenceTasks_TrainingID`         ON `InferenceTasks`(`TrainingID`);
CREATE INDEX IF NOT EXISTS `IX_InferenceResults_TaskID`           ON `InferenceResults`(`InferenceTaskID`);
CREATE INDEX IF NOT EXISTS `IX_InferenceResults_Timestamp`        ON `InferenceResults`(`ResultTimestamp`);

/* ---------------- 初始数据（与 MySQL 版一致） ---------------- */
INSERT OR IGNORE INTO `Datasets` (`DatasetName`, `Source`, `SampleCount`, `ClassCount`, `DataPath`, `Description`)
VALUES ('CWRU-0HP', 'Case Western Reserve University 轴承数据集', NULL, 10,
        'testRestfulProject/1DCNN/0HP', '驱动端(DE)振动信号，10 类轴承状态');

INSERT OR IGNORE INTO `Models` (`ModelName`, `Description`, `ApiEndpoint`, `ModelType`, `Status`)
VALUES
 ('1DCNN',   '一维卷积神经网络，CWRU 轴承振动信号 10 类故障分类。', '/predict', 'Classification',     '可运行'),
 ('cwt_cnn', '与 1DCNN 同任务的 PyTorch 实现，输出混淆矩阵。',      '/predict', 'Classification',     '可运行'),
 ('adtk',    '时序异常检测库（无监督），项目经 main.py 调用 PcaAD。', '/predict', 'AnomalyDetection',   '可运行');
