/* ================================================================
   模型管理库 · MySQL 版 建表脚本
   ----------------------------------------------------------------
   目标数据库: model_management
   字符集: utf8mb4 / utf8mb4_unicode_ci
   引擎: InnoDB
   表清单:
     1. Datasets            数据集
     2. Models              模型
     3. EdgeDevices         边缘设备
     4. Trainings           训练记录
     5. ModelInvocations    模型调用日志
     6. ModelDeployments    模型发布记录
     7. InferenceTasks      推理任务
     8. InferenceResults    推理结果明细
   特点:
     - CREATE TABLE IF NOT EXISTS，可重复执行
     - 外键在建表时内联（依赖顺序已排好）
     - 种子数据用 INSERT IGNORE，依赖唯一键去重
   T-SQL → MySQL 映射:
     [int] IDENTITY(1,1)   -> INT AUTO_INCREMENT
     [nvarchar](n)         -> VARCHAR(n)
     [nvarchar](max)       -> LONGTEXT
     [datetime2](7)        -> DATETIME(6)
     [bit]                 -> TINYINT(1)
     [float]               -> DOUBLE
     SYSDATETIME()         -> CURRENT_TIMESTAMP(6)
   ================================================================ */

CREATE DATABASE IF NOT EXISTS `model_management`
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `model_management`;


/* ---------------- 1. Datasets ---------------- */
CREATE TABLE IF NOT EXISTS `Datasets` (
    `DatasetID`    INT           NOT NULL AUTO_INCREMENT,
    `DatasetName`  VARCHAR(100)  NOT NULL,
    `Source`       VARCHAR(200)  NULL,
    `SampleCount`  INT           NULL,
    `ClassCount`   INT           NULL,
    `DataPath`     VARCHAR(500)  NULL,
    `Description`  VARCHAR(500)  NULL,
    `CreatedDate`  DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`DatasetID`),
    UNIQUE KEY `UQ_Datasets_DatasetName` (`DatasetName`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 2. Models ---------------- */
CREATE TABLE IF NOT EXISTS `Models` (
    `ModelID`     INT           NOT NULL AUTO_INCREMENT,
    `ModelName`   VARCHAR(100)  NOT NULL,
    `Description` VARCHAR(500)  NULL,
    `ApiEndpoint` VARCHAR(255)  NULL,
    `ModelType`   VARCHAR(50)   NULL,
    `CreatedDate` DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    `IsActive`    TINYINT(1)    NULL DEFAULT 1,
    `Status`      VARCHAR(20)   NULL,
    PRIMARY KEY (`ModelID`),
    UNIQUE KEY `UQ_Models_ModelName` (`ModelName`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 3. EdgeDevices ---------------- */
CREATE TABLE IF NOT EXISTS `EdgeDevices` (
    `DeviceID`      INT           NOT NULL AUTO_INCREMENT,
    `DeviceName`    VARCHAR(100)  NOT NULL,
    `DeviceType`    VARCHAR(50)   NULL,
    `IPAddress`     VARCHAR(45)   NULL,
    `Location`      VARCHAR(255)  NULL,
    `HardwareSpecs` LONGTEXT      NULL,
    `EdgeStatus`    VARCHAR(20)   NULL,
    `LastHeartbeat` DATETIME(6)   NULL,
    `DeviceCode`    VARCHAR(100)  NULL,
    `MacAddress`    VARCHAR(50)   NULL,
    `OsVersion`     VARCHAR(100)  NULL,
    `Status`        VARCHAR(20)   NULL,
    `IsActive`      TINYINT(1)    NULL DEFAULT 1,
    `Remark`        VARCHAR(500)  NULL,
    `CreatedDate`   DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`DeviceID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 4. Trainings ---------------- */
CREATE TABLE IF NOT EXISTS `Trainings` (
    `TrainingID`    INT           NOT NULL AUTO_INCREMENT,
    `ModelID`       INT           NOT NULL,
    `DatasetID`     INT           NULL,
    `TrainName`     VARCHAR(200)  NULL,
    `Epochs`        INT           NULL,
    `BatchSize`     INT           NULL,
    `Accuracy`      DOUBLE        NULL,
    `Loss`          DOUBLE        NULL,
    `ModelPath`     VARCHAR(500)  NULL,
    `Status`        VARCHAR(20)   NULL,
    `CreatedDate`   DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    `StartedDate`   DATETIME(6)   NULL,
    `CompletedDate` DATETIME(6)   NULL,
    `CreatedBy`     VARCHAR(100)  NULL,
    `Remark`        VARCHAR(500)  NULL,
    PRIMARY KEY (`TrainingID`),
    KEY `IX_Trainings_ModelID` (`ModelID`),
    CONSTRAINT `FK_Trainings_Models`   FOREIGN KEY (`ModelID`)   REFERENCES `Models` (`ModelID`),
    CONSTRAINT `FK_Trainings_Datasets` FOREIGN KEY (`DatasetID`) REFERENCES `Datasets` (`DatasetID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 5. ModelInvocations ---------------- */
CREATE TABLE IF NOT EXISTS `ModelInvocations` (
    `InvocationID`   INT           NOT NULL AUTO_INCREMENT,
    `ModelID`        INT           NOT NULL,
    `TrainingID`     INT           NULL,
    `ApiEndpoint`    VARCHAR(255)  NULL,
    `RequestParams`  LONGTEXT      NOT NULL,
    `ResponseResult` LONGTEXT      NULL,
    `DurationMs`     INT           NULL,
    `IsSuccess`      TINYINT(1)    NULL,
    `StatusCode`     INT           NULL,
    `ErrorMessage`   LONGTEXT      NULL,
    `ClientIP`       VARCHAR(50)   NULL,
    `Status`         VARCHAR(20)   NULL,
    `InvocationDate` DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`InvocationID`),
    KEY `IX_ModelInvocations_ModelID_Date` (`ModelID`, `InvocationDate`),
    CONSTRAINT `FK_ModelInvocations_Models`    FOREIGN KEY (`ModelID`)    REFERENCES `Models` (`ModelID`),
    CONSTRAINT `FK_ModelInvocations_Trainings` FOREIGN KEY (`TrainingID`) REFERENCES `Trainings` (`TrainingID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 6. ModelDeployments ---------------- */
CREATE TABLE IF NOT EXISTS `ModelDeployments` (
    `DeploymentID`    INT           NOT NULL AUTO_INCREMENT,
    `ModelID`         INT           NOT NULL,
    `TrainingID`      INT           NOT NULL,
    `DeviceID`        INT           NOT NULL,
    `Version`         VARCHAR(50)   NULL,
    `VersionAlias`    VARCHAR(20)   NULL,
    `Environment`     VARCHAR(20)   NULL,
    `DeployUrl`       VARCHAR(255)  NULL,
    `DeployedPath`    VARCHAR(500)  NULL,
    `ServicePort`     INT           NULL,
    `RuntimeParams`   LONGTEXT      NULL,
    `IsActive`        TINYINT(1)    NULL DEFAULT 1,
    `IsCurrent`       TINYINT(1)    NOT NULL DEFAULT 1,
    `DeployStatus`    VARCHAR(20)   NULL,
    `LastStatusCheck` DATETIME(6)   NULL,
    `DeployedDate`    DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    `DeployedBy`      VARCHAR(100)  NULL,
    `ErrorMessage`    LONGTEXT      NULL,
    `Remark`          VARCHAR(500)  NULL,
    PRIMARY KEY (`DeploymentID`),
    KEY `IX_ModelDeployments_ModelID_Current` (`ModelID`, `IsCurrent`),
    KEY `IX_ModelDeployments_DeviceID` (`DeviceID`),
    CONSTRAINT `FK_ModelDeployments_Models`      FOREIGN KEY (`ModelID`)    REFERENCES `Models` (`ModelID`),
    CONSTRAINT `FK_ModelDeployments_Trainings`   FOREIGN KEY (`TrainingID`) REFERENCES `Trainings` (`TrainingID`),
    CONSTRAINT `FK_ModelDeployments_EdgeDevices` FOREIGN KEY (`DeviceID`)   REFERENCES `EdgeDevices` (`DeviceID`),
    CONSTRAINT `CK_ModelDeployments_ServicePort` CHECK (`ServicePort` IS NULL OR (`ServicePort` BETWEEN 1 AND 65535))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 7. InferenceTasks ---------------- */
CREATE TABLE IF NOT EXISTS `InferenceTasks` (
    `InferenceTaskID`  INT           NOT NULL AUTO_INCREMENT,
    `TrainingID`       INT           NOT NULL,
    `TargetDatasetID`  INT           NOT NULL,
    `TaskName`         VARCHAR(200)  NOT NULL,
    `TaskType`         VARCHAR(20)   NOT NULL,
    `Status`           VARCHAR(20)   NULL,
    `InferenceParams`  LONGTEXT      NULL,
    `ResultSummary`    LONGTEXT      NULL,
    `ErrorMessage`     LONGTEXT      NULL,
    `DeploymentID`     INT           NULL,
    `ModelID`          INT           NULL,
    `DeviceID`         INT           NULL,
    `InputPath`        VARCHAR(500)  NULL,
    `OutputPath`       VARCHAR(500)  NULL,
    `Progress`         INT           NULL,
    `CreatedDate`      DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    `StartedDate`      DATETIME(6)   NULL,
    `CompletedDate`    DATETIME(6)   NULL,
    `CreatedBy`        VARCHAR(100)  NULL,
    PRIMARY KEY (`InferenceTaskID`),
    KEY `IX_InferenceTasks_Status` (`Status`),
    KEY `IX_InferenceTasks_TrainingID` (`TrainingID`),
    CONSTRAINT `FK_InferenceTasks_Trainings`   FOREIGN KEY (`TrainingID`)      REFERENCES `Trainings` (`TrainingID`),
    CONSTRAINT `FK_InferenceTasks_Datasets`    FOREIGN KEY (`TargetDatasetID`) REFERENCES `Datasets` (`DatasetID`),
    CONSTRAINT `FK_InferenceTasks_Models`      FOREIGN KEY (`ModelID`)         REFERENCES `Models` (`ModelID`),
    CONSTRAINT `FK_InferenceTasks_EdgeDevices` FOREIGN KEY (`DeviceID`)        REFERENCES `EdgeDevices` (`DeviceID`),
    CONSTRAINT `FK_InferenceTasks_Deployments` FOREIGN KEY (`DeploymentID`)    REFERENCES `ModelDeployments` (`DeploymentID`),
    CONSTRAINT `CK_InferenceTasks_Progress` CHECK (`Progress` IS NULL OR (`Progress` BETWEEN 0 AND 100))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 8. InferenceResults ---------------- */
CREATE TABLE IF NOT EXISTS `InferenceResults` (
    `ResultID`          BIGINT        NOT NULL AUTO_INCREMENT,
    `InferenceTaskID`   INT           NOT NULL,
    `RowIdentifier`     VARCHAR(100)  NULL,
    `ResultTimestamp`   DATETIME(6)   NULL,
    `PredictedValue`    DOUBLE        NULL,
    `AnomalyScore`      DOUBLE        NULL,
    `IsAnomaly`         TINYINT(1)    NULL,
    `PredictedCategory` VARCHAR(50)   NULL,
    `Confidence`        DOUBLE        NULL,
    `FeatureSnapshot`   VARCHAR(500)  NULL,
    `ModelID`           INT           NULL,
    `SampleIndex`       INT           NULL,
    `PredictedClass`    INT           NULL,
    `PredictedLabel`    VARCHAR(100)  NULL,
    `Score`             DOUBLE        NULL,
    `ActualClass`       INT           NULL,
    `ResultDetail`      LONGTEXT      NULL,
    `CreatedDate`       DATETIME(6)   NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (`ResultID`),
    KEY `IX_InferenceResults_TaskID` (`InferenceTaskID`),
    KEY `IX_InferenceResults_Timestamp` (`ResultTimestamp`),
    CONSTRAINT `FK_InferenceResults_Tasks`  FOREIGN KEY (`InferenceTaskID`) REFERENCES `InferenceTasks` (`InferenceTaskID`),
    CONSTRAINT `FK_InferenceResults_Models` FOREIGN KEY (`ModelID`)          REFERENCES `Models` (`ModelID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


/* ---------------- 初始数据 ---------------- */
INSERT IGNORE INTO `Datasets` (`DatasetName`, `Source`, `SampleCount`, `ClassCount`, `DataPath`, `Description`)
VALUES ('CWRU-0HP', 'Case Western Reserve University 轴承数据集', NULL, 10,
        'testRestfulProject/1DCNN/0HP', '驱动端(DE)振动信号，10 类轴承状态');

INSERT IGNORE INTO `Models` (`ModelName`, `Description`, `ApiEndpoint`, `ModelType`, `Status`)
VALUES
 ('1DCNN',   '一维卷积神经网络，CWRU 轴承振动信号 10 类故障分类。', NULL, 'Classification',   '可运行'),
 ('cwt_cnn', '与 1DCNN 同任务的 PyTorch 实现，输出混淆矩阵。',       NULL, 'Classification',   '可运行'),
 ('adtk',    '时序异常检测库（无监督），项目经 main.py 调用 PcaAD。', '/todos', 'AnomalyDetection', '可运行');
