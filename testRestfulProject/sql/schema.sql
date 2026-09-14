/* ================================================================
   模型管理库 · 建表脚本  (Microsoft SQL Server / T-SQL)
   ----------------------------------------------------------------
   用途：登记并追踪「模型 → 训练 → 部署 → 推理」全生命周期
   表清单（按外键依赖顺序）：
     1. Datasets            数据集
     2. Models              模型
     3. EdgeDevices         边缘设备
     4. Trainings           训练记录
     5. ModelInvocations    模型调用日志
     6. ModelDeployments    模型发布记录
     7. InferenceTasks      推理任务
     8. InferenceResults    推理结果明细
   说明：
     - 所有表用 IF OBJECT_ID ... IS NULL 保护，可重复执行
     - 外键、唯一约束、索引集中在末尾追加（带存在性判断）
     - 列的可空性与原始设计保持一致；仅给时间/Bool 列补了默认值
   ================================================================ */

SET ANSI_NULLS ON;
GO
SET QUOTED_IDENTIFIER ON;
GO


/* ================================================================
   1. Datasets —— 数据集
   ================================================================ */
IF OBJECT_ID(N'[dbo].[Datasets]', N'U') IS NULL
CREATE TABLE [dbo].[Datasets](
    /* 数据集：DatasetName 唯一，db.ensure_dataset() 按它做"有就取、没有就建" */
    [DatasetID]    [int]           IDENTITY(1,1) NOT NULL,
    [DatasetName]  [nvarchar](100) NOT NULL,
    [Source]       [nvarchar](200) NULL,      -- CWRU 官方数据集 / 用户上传 / ADHOC 临时登记
    [SampleCount]  [int]           NULL,      -- 样本（窗口）数
    [ClassCount]   [int]           NULL,      -- 类别数；无监督模型为 NULL
    [DataPath]     [nvarchar](500) NULL,      -- 数据目录（接口返回时已脱敏）
    [Description]  [nvarchar](500) NULL,
    [CreatedDate]  [datetime2](7)  NULL CONSTRAINT [DF_Datasets_CreatedDate] DEFAULT (SYSDATETIME()),
    CONSTRAINT [PK_Datasets] PRIMARY KEY CLUSTERED ([DatasetID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   2. Models —— 模型
   ================================================================ */
IF OBJECT_ID(N'[dbo].[Models]', N'U') IS NULL
CREATE TABLE [dbo].[Models](
    /* 模型登记表：ModelName 唯一；训练/上传时按名字"有就取用、不重复插" */
    [ModelID]     [int]           IDENTITY(1,1) NOT NULL,
    [ModelName]   [nvarchar](100) NOT NULL,
    [Description] [nvarchar](500) NULL,
    [ApiEndpoint] [nvarchar](255) NULL,       -- 该模型的调用入口，一般为 /predict
    [ModelType]   [nvarchar](50)  NULL,       -- Classification / AnomalyDetection / Regression
    [CreatedDate] [datetime2](7)  NULL CONSTRAINT [DF_Models_CreatedDate] DEFAULT (SYSDATETIME()),
    [IsActive]    [bit]           NULL CONSTRAINT [DF_Models_IsActive] DEFAULT ((1)),
    [Status]      [nvarchar](20)  NULL,       -- 可运行 / 未训练 …
    CONSTRAINT [PK_Models] PRIMARY KEY CLUSTERED ([ModelID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   3. EdgeDevices —— 边缘设备
   ================================================================ */
IF OBJECT_ID(N'[dbo].[EdgeDevices]', N'U') IS NULL
CREATE TABLE [dbo].[EdgeDevices](
    /* 边缘设备：当前 model_service 不写这张表（「边缘设备」支线未落地），保留结构供后续部署使用 */
    [DeviceID]      [int]           IDENTITY(1,1) NOT NULL,
    [DeviceName]    [nvarchar](100) NOT NULL,
    [DeviceType]    [nvarchar](50)  NULL,
    [IPAddress]     [nvarchar](45)  NULL,     -- IPv4/IPv6 文本
    [Location]      [nvarchar](255) NULL,
    [HardwareSpecs] [nvarchar](max) NULL,     -- 硬件规格（自由文本 / JSON）
    [EdgeStatus]    [nvarchar](20)  NULL,     -- 设备侧运行状态（与 Status 语义重叠，见文件末尾备注）
    [LastHeartbeat] [datetime2](7)  NULL,
    [DeviceCode]    [nvarchar](100) NULL,
    [MacAddress]    [nvarchar](50)  NULL,
    [OsVersion]     [nvarchar](100) NULL,
    [Status]        [nvarchar](20)  NULL,     -- 本平台的登记状态
    [IsActive]      [bit]           NULL CONSTRAINT [DF_EdgeDevices_IsActive] DEFAULT ((1)),
    [Remark]        [nvarchar](500) NULL,
    [CreatedDate]   [datetime2](7)  NULL CONSTRAINT [DF_EdgeDevices_CreatedDate] DEFAULT (SYSDATETIME()),
    CONSTRAINT [PK_EdgeDevices] PRIMARY KEY CLUSTERED ([DeviceID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   4. Trainings —— 训练记录（被 3 张表引用，是这套结构的中枢）
   ================================================================ */
IF OBJECT_ID(N'[dbo].[Trainings]', N'U') IS NULL
CREATE TABLE [dbo].[Trainings](
    /* 训练记录：被 ModelInvocations / ModelDeployments / InferenceTasks 三张表引用，
       是整套结构的中枢——InferenceTasks.TrainingID 就是这里的"权威锚点" */
    [TrainingID]    [int]           IDENTITY(1,1) NOT NULL,
    [ModelID]       [int]           NOT NULL,   -- 外键，模型必须已登记
    [DatasetID]     [int]           NULL,       -- 可空：删除数据集时会置空
    [TrainName]     [nvarchar](200) NULL,       -- 形如 1dcnn-20250912-193000
    [Epochs]        [int]           NULL,
    [BatchSize]     [int]           NULL,
    [Accuracy]      [float]         NULL,       -- 测试集准确率；无监督模型为 NULL
    [Loss]          [float]         NULL,
    [ModelPath]     [nvarchar](500) NULL,       -- 权重文件路径，指向 data/models/<模型>/vN/…
    [Status]        [nvarchar](20)  NULL,       -- 成功 / 失败
    [CreatedDate]   [datetime2](7)  NULL CONSTRAINT [DF_Trainings_CreatedDate] DEFAULT (SYSDATETIME()),
    [StartedDate]   [datetime2](7)  NULL,
    [CompletedDate] [datetime2](7)  NULL,
    [CreatedBy]     [nvarchar](100) NULL,
    [Remark]        [nvarchar](500) NULL,       -- 跳过越界窗口数、NaN 窗口数、日志/图目录等 JSON
    CONSTRAINT [PK_Trainings] PRIMARY KEY CLUSTERED ([TrainingID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   5. ModelInvocations —— 模型调用日志
   ================================================================ */
IF OBJECT_ID(N'[dbo].[ModelInvocations]', N'U') IS NULL
CREATE TABLE [dbo].[ModelInvocations](
    /* 每次 /predict 留一条（失败也留）——最简单的调用审计日志 */
    [InvocationID]   [int]           IDENTITY(1,1) NOT NULL,
    [ModelID]        [int]           NOT NULL,
    [TrainingID]     [int]           NULL,      -- 能定位到哪次训练时才有值
    [ApiEndpoint]    [nvarchar](255) NULL,      -- 目前都是 /predict
    [RequestParams]  [nvarchar](max) NOT NULL,  -- 请求参数 JSON（samples 体积太大，不记）
    [ResponseResult] [nvarchar](max) NULL,      -- 成功摘要 / 失败错误 JSON
    [DurationMs]     [int]           NULL,      -- 本次耗时（毫秒）
    [IsSuccess]      [bit]           NULL,      -- 1/0
    [StatusCode]     [int]           NULL,      -- HTTP 状态码
    [ErrorMessage]   [nvarchar](max) NULL,
    [ClientIP]       [nvarchar](50)  NULL,
    [Status]         [nvarchar](20)  NULL,      -- 成功 / 失败
    [InvocationDate] [datetime2](7)  NULL CONSTRAINT [DF_ModelInvocations_InvocationDate] DEFAULT (SYSDATETIME()),
    CONSTRAINT [PK_ModelInvocations] PRIMARY KEY CLUSTERED ([InvocationID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   6. ModelDeployments —— 模型发布记录
   ================================================================ */
IF OBJECT_ID(N'[dbo].[ModelDeployments]', N'U') IS NULL
CREATE TABLE [dbo].[ModelDeployments](
    /* 模型发布记录（把某次训练产出的模型投放到某台边缘设备）。当前服务同样不写这张表 */
    [DeploymentID]    [int]           IDENTITY(1,1) NOT NULL,
    [ModelID]         [int]           NOT NULL,
    [TrainingID]      [int]           NOT NULL,   -- 发布的是哪一次训练的产物
    [DeviceID]        [int]           NOT NULL,
    [Version]         [nvarchar](50)  NULL,       -- 版本号，如 v2
    [VersionAlias]    [nvarchar](20)  NULL,       -- 别名，如 latest / stable
    [Environment]     [nvarchar](20)  NULL,       -- 环境：prod / test …
    [DeployUrl]       [nvarchar](255) NULL,       -- 部署后的访问地址
    [DeployedPath]    [nvarchar](500) NULL,       -- 设备上的产物路径
    [ServicePort]     [int]           NULL,
    [RuntimeParams]   [nvarchar](max) NULL,       -- 运行参数（JSON）
    [IsActive]        [bit]           NULL CONSTRAINT [DF_ModelDeployments_IsActive] DEFAULT ((1)),
    [IsCurrent]       [bit]           NOT NULL CONSTRAINT [DF_ModelDeployments_IsCurrent] DEFAULT ((1)),
    [DeployStatus]    [nvarchar](20)  NULL,
    [LastStatusCheck] [datetime2](7)  NULL,
    [DeployedDate]    [datetime2](7)  NULL CONSTRAINT [DF_ModelDeployments_DeployedDate] DEFAULT (SYSDATETIME()),
    [DeployedBy]      [nvarchar](100) NULL,
    [ErrorMessage]    [nvarchar](max) NULL,
    [Remark]          [nvarchar](500) NULL,
    CONSTRAINT [PK_ModelDeployments] PRIMARY KEY CLUSTERED ([DeploymentID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   7. InferenceTasks —— 推理任务
   ================================================================ */
IF OBJECT_ID(N'[dbo].[InferenceTasks]', N'U') IS NULL
CREATE TABLE [dbo].[InferenceTasks](
    /* 一次推理 = 一个任务，明细在 InferenceResults。
       TrainingID / TargetDatasetID 都是 NOT NULL 外键：推理前必须已有成功的训练记录 */
    [InferenceTaskID]  [int]           IDENTITY(1,1) NOT NULL,
    [TrainingID]       [int]           NOT NULL,   -- 权威锚点
    [TargetDatasetID]  [int]           NOT NULL,   -- 内联样本会登记成 ADHOC-<模型> 数据集
    [TaskName]         [nvarchar](200) NOT NULL,   -- 形如 predict-1dcnn-20250912-193000
    [TaskType]         [nvarchar](20)  NOT NULL,   -- classification / anomaly_detection
    [Status]           [nvarchar](20)  NULL,
    [InferenceParams]  [nvarchar](max) NULL,       -- 请求参数 JSON
    [ResultSummary]    [nvarchar](max) NULL,       -- 样本数、预测分布、模型版本…
    [ErrorMessage]     [nvarchar](max) NULL,
    [DeploymentID]     [int]           NULL,       -- 当前留空，等边缘设备支线落地后回填
    [ModelID]          [int]           NULL,
    [DeviceID]         [int]           NULL,       -- 同上，留空
    [InputPath]        [nvarchar](500) NULL,       -- 输入来源文件路径
    [OutputPath]       [nvarchar](500) NULL,       -- 出图目录
    [Progress]         [int]           NULL,
    [CreatedDate]      [datetime2](7)  NULL CONSTRAINT [DF_InferenceTasks_CreatedDate] DEFAULT (SYSDATETIME()),
    [StartedDate]      [datetime2](7)  NULL,
    [CompletedDate]    [datetime2](7)  NULL,
    [CreatedBy]        [nvarchar](100) NULL,
    CONSTRAINT [PK_InferenceTasks] PRIMARY KEY CLUSTERED ([InferenceTaskID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   8. InferenceResults —— 推理结果明细（1 任务 : N 结果）
   ================================================================ */
IF OBJECT_ID(N'[dbo].[InferenceResults]', N'U') IS NULL
CREATE TABLE [dbo].[InferenceResults](
    /* 推理结果明细，一个任务 N 行（一个窗口一行）。
       分类任务用 PredictedClass/Label/Confidence，异常检测用 IsAnomaly/AnomalyScore，
       两类共享同一张表，所以近义列较多（见文件末尾的冗余列备注） */
    [ResultID]          [bigint]        IDENTITY(1,1) NOT NULL,
    [InferenceTaskID]   [int]           NOT NULL,
    [RowIdentifier]     [nvarchar](100) NULL,   -- 形如 内圈故障.csv#3，用来回溯这一行的来源
    [ResultTimestamp]   [datetime2](7)  NULL,   -- 该行结果的时间（取写入时刻）
    [PredictedValue]    [float]         NULL,   -- 分类任务里的置信度（同类名字段）
    [AnomalyScore]      [float]         NULL,   -- 异常检测：窗口重构误差 / 异常点占比
    [IsAnomaly]         [bit]           NULL,
    [PredictedCategory] [nvarchar](50)  NULL,   -- Classification / AnomalyDetection
    [Confidence]        [float]         NULL,
    [FeatureSnapshot]   [nvarchar](500) NULL,   -- top_k 的 JSON 快照（超 500 字符只记 truncated）
    [ModelID]           [int]           NULL,
    [SampleIndex]       [int]           NULL,   -- 窗口序号
    [PredictedClass]    [int]           NULL,   -- 类别号
    [PredictedLabel]    [nvarchar](100) NULL,   -- 类别中文名
    [Score]             [float]         NULL,   -- 与 Confidence 同值
    [ActualClass]       [int]           NULL,   -- 输入来自登记过的数据集文件时才有真值
    [ResultDetail]      [nvarchar](max) NULL,   -- 判定细节 JSON（阈值、重构误差、相对倍数…）
    [CreatedDate]       [datetime2](7)  NULL CONSTRAINT [DF_InferenceResults_CreatedDate] DEFAULT (SYSDATETIME()),
    CONSTRAINT [PK_InferenceResults] PRIMARY KEY CLUSTERED ([ResultID] ASC)
) ON [PRIMARY];
GO


/* ================================================================
   外键约束
   ================================================================ */

/* Trainings */
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_Trainings_Models')
ALTER TABLE [dbo].[Trainings] ADD CONSTRAINT [FK_Trainings_Models]
    FOREIGN KEY([ModelID]) REFERENCES [dbo].[Models]([ModelID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_Trainings_Datasets')
ALTER TABLE [dbo].[Trainings] ADD CONSTRAINT [FK_Trainings_Datasets]
    FOREIGN KEY([DatasetID]) REFERENCES [dbo].[Datasets]([DatasetID]);
GO

/* ModelInvocations */
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_ModelInvocations_Models')
ALTER TABLE [dbo].[ModelInvocations] ADD CONSTRAINT [FK_ModelInvocations_Models]
    FOREIGN KEY([ModelID]) REFERENCES [dbo].[Models]([ModelID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_ModelInvocations_Trainings')
ALTER TABLE [dbo].[ModelInvocations] ADD CONSTRAINT [FK_ModelInvocations_Trainings]
    FOREIGN KEY([TrainingID]) REFERENCES [dbo].[Trainings]([TrainingID]);
GO

/* ModelDeployments */
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_ModelDeployments_Models')
ALTER TABLE [dbo].[ModelDeployments] ADD CONSTRAINT [FK_ModelDeployments_Models]
    FOREIGN KEY([ModelID]) REFERENCES [dbo].[Models]([ModelID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_ModelDeployments_Trainings')
ALTER TABLE [dbo].[ModelDeployments] ADD CONSTRAINT [FK_ModelDeployments_Trainings]
    FOREIGN KEY([TrainingID]) REFERENCES [dbo].[Trainings]([TrainingID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_ModelDeployments_EdgeDevices')
ALTER TABLE [dbo].[ModelDeployments] ADD CONSTRAINT [FK_ModelDeployments_EdgeDevices]
    FOREIGN KEY([DeviceID]) REFERENCES [dbo].[EdgeDevices]([DeviceID]);
GO

/* InferenceTasks */
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_InferenceTasks_Trainings')
ALTER TABLE [dbo].[InferenceTasks] ADD CONSTRAINT [FK_InferenceTasks_Trainings]
    FOREIGN KEY([TrainingID]) REFERENCES [dbo].[Trainings]([TrainingID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_InferenceTasks_Datasets')
ALTER TABLE [dbo].[InferenceTasks] ADD CONSTRAINT [FK_InferenceTasks_Datasets]
    FOREIGN KEY([TargetDatasetID]) REFERENCES [dbo].[Datasets]([DatasetID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_InferenceTasks_Models')
ALTER TABLE [dbo].[InferenceTasks] ADD CONSTRAINT [FK_InferenceTasks_Models]
    FOREIGN KEY([ModelID]) REFERENCES [dbo].[Models]([ModelID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_InferenceTasks_EdgeDevices')
ALTER TABLE [dbo].[InferenceTasks] ADD CONSTRAINT [FK_InferenceTasks_EdgeDevices]
    FOREIGN KEY([DeviceID]) REFERENCES [dbo].[EdgeDevices]([DeviceID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_InferenceTasks_Deployments')
ALTER TABLE [dbo].[InferenceTasks] ADD CONSTRAINT [FK_InferenceTasks_Deployments]
    FOREIGN KEY([DeploymentID]) REFERENCES [dbo].[ModelDeployments]([DeploymentID]);
GO

/* InferenceResults */
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_InferenceResults_Tasks')
ALTER TABLE [dbo].[InferenceResults] ADD CONSTRAINT [FK_InferenceResults_Tasks]
    FOREIGN KEY([InferenceTaskID]) REFERENCES [dbo].[InferenceTasks]([InferenceTaskID]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = N'FK_InferenceResults_Models')
ALTER TABLE [dbo].[InferenceResults] ADD CONSTRAINT [FK_InferenceResults_Models]
    FOREIGN KEY([ModelID]) REFERENCES [dbo].[Models]([ModelID]);
GO


/* ================================================================
   检查约束（可选，按需保留）
   ================================================================ */
IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name = N'CK_InferenceTasks_Progress')
ALTER TABLE [dbo].[InferenceTasks] WITH NOCHECK ADD CONSTRAINT [CK_InferenceTasks_Progress]
    CHECK ([Progress] IS NULL OR [Progress] BETWEEN 0 AND 100);
GO
IF NOT EXISTS (SELECT 1 FROM sys.check_constraints WHERE name = N'CK_ModelDeployments_ServicePort')
ALTER TABLE [dbo].[ModelDeployments] WITH NOCHECK ADD CONSTRAINT [CK_ModelDeployments_ServicePort]
    CHECK ([ServicePort] IS NULL OR [ServicePort] BETWEEN 1 AND 65535);
GO


/* ================================================================
   唯一约束（可选）
   ================================================================ */
IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = N'UQ_Models_ModelName')
ALTER TABLE [dbo].[Models] ADD CONSTRAINT [UQ_Models_ModelName] UNIQUE ([ModelName]);
GO
IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = N'UQ_Datasets_DatasetName')
ALTER TABLE [dbo].[Datasets] ADD CONSTRAINT [UQ_Datasets_DatasetName] UNIQUE ([DatasetName]);
GO


/* ================================================================
   索引
   ================================================================ */
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_Trainings_ModelID' AND object_id = OBJECT_ID(N'[dbo].[Trainings]'))
CREATE NONCLUSTERED INDEX [IX_Trainings_ModelID] ON [dbo].[Trainings]([ModelID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_ModelInvocations_ModelID_Date' AND object_id = OBJECT_ID(N'[dbo].[ModelInvocations]'))
CREATE NONCLUSTERED INDEX [IX_ModelInvocations_ModelID_Date] ON [dbo].[ModelInvocations]([ModelID] ASC, [InvocationDate] DESC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_ModelDeployments_ModelID_Current' AND object_id = OBJECT_ID(N'[dbo].[ModelDeployments]'))
CREATE NONCLUSTERED INDEX [IX_ModelDeployments_ModelID_Current] ON [dbo].[ModelDeployments]([ModelID] ASC, [IsCurrent] ASC);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_ModelDeployments_DeviceID' AND object_id = OBJECT_ID(N'[dbo].[ModelDeployments]'))
CREATE NONCLUSTERED INDEX [IX_ModelDeployments_DeviceID] ON [dbo].[ModelDeployments]([DeviceID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_InferenceTasks_Status' AND object_id = OBJECT_ID(N'[dbo].[InferenceTasks]'))
CREATE NONCLUSTERED INDEX [IX_InferenceTasks_Status] ON [dbo].[InferenceTasks]([Status] ASC);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_InferenceTasks_TrainingID' AND object_id = OBJECT_ID(N'[dbo].[InferenceTasks]'))
CREATE NONCLUSTERED INDEX [IX_InferenceTasks_TrainingID] ON [dbo].[InferenceTasks]([TrainingID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_InferenceResults_TaskID' AND object_id = OBJECT_ID(N'[dbo].[InferenceResults]'))
CREATE NONCLUSTERED INDEX [IX_InferenceResults_TaskID] ON [dbo].[InferenceResults]([InferenceTaskID] ASC);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = N'IX_InferenceResults_Timestamp' AND object_id = OBJECT_ID(N'[dbo].[InferenceResults]'))
CREATE NONCLUSTERED INDEX [IX_InferenceResults_Timestamp] ON [dbo].[InferenceResults]([ResultTimestamp] ASC);
GO


/* ================================================================
   初始数据（可重复执行，按名称判重）
   ================================================================ */

/* 数据集：CWRU 轴承 */
IF NOT EXISTS (SELECT 1 FROM [dbo].[Datasets] WHERE [DatasetName] = N'CWRU-0HP')
INSERT INTO [dbo].[Datasets] ([DatasetName], [Source], [SampleCount], [ClassCount], [DataPath], [Description])
VALUES (N'CWRU-0HP', N'Case Western Reserve University 轴承数据集', NULL, 10,
        N'testRestfulProject\1DCNN\0HP', N'驱动端(DE)振动信号，10 类轴承状态');
GO

/* 模型：项目内的三个 */
IF NOT EXISTS (SELECT 1 FROM [dbo].[Models] WHERE [ModelName] = N'1DCNN')
INSERT INTO [dbo].[Models] ([ModelName], [Description], [ApiEndpoint], [ModelType], [Status])
VALUES (N'1DCNN', N'一维卷积神经网络，CWRU 轴承振动信号 10 类故障分类。', NULL, N'Classification', N'可运行');
GO
IF NOT EXISTS (SELECT 1 FROM [dbo].[Models] WHERE [ModelName] = N'cwt_cnn')
INSERT INTO [dbo].[Models] ([ModelName], [Description], [ApiEndpoint], [ModelType], [Status])
VALUES (N'cwt_cnn', N'与 1DCNN 同任务的 PyTorch 实现，输出混淆矩阵。', NULL, N'Classification', N'可运行');
GO
IF NOT EXISTS (SELECT 1 FROM [dbo].[Models] WHERE [ModelName] = N'adtk')
INSERT INTO [dbo].[Models] ([ModelName], [Description], [ApiEndpoint], [ModelType], [Status])
VALUES (N'adtk', N'时序异常检测库（无监督），项目经 main.py 调用 PcaAD。', N'/todos', N'AnomalyDetection', N'可运行');
GO


/* ================================================================
   备注：已知的冗余列（未自动删除，需人工决定去留）
   ----------------------------------------------------------------
   Models            : Status 与 IsActive 语义重叠
   ModelDeployments  : IsActive / IsCurrent / DeployStatus 三者需分工
   InferenceResults  : PredictedClass / PredictedLabel / PredictedCategory 近义
                       Score / AnomalyScore / PredictedValue 近义
   EdgeDevices       : EdgeStatus 与 Status 重名语义
   InferenceTasks    : TrainingID 与 DeploymentID 同时存在，需明确权威锚点
   ================================================================ */
