# model_service —— 把流程图上的「训练 → 模型产物 → 推理」接起来

对应你画的那张流程图：

```
Web服务器 ─▶ flask_restful 接口(Web访问) ─┬─▶ 算法模型1(1DCNN)  ──训练──▶ 数据集
                                          ├─▶ 算法模型2(cwt_cnn)          │
                                          └─▶ 算法模型3(adtk)             ▼
                                   边缘设备 ◀── 推理 ◀── Pxl模型(模型产物)
```

改造前这条链是**断的**：两个训练脚本跑完就把模型丢在内存里，仓库里没有任何权重文件，
`Trainings.ModelPath` 无值可填，也没有任何推理代码，SQL 里那 8 张表更是一行都没被写过。
本模块把断点补上，并且**每一环都能被验证**。

---

## 一、接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 接口索引（等同 `/api`；原先 302 跳到已删除的 `/ui` 控制台） |
| GET | `/api` | 接口索引 |
| GET | `/health` | 服务 / 数据库 / 产物体检 |
| GET | `/models` | 落盘产物 + 库表登记的模型清单 |
| GET | `/models/<name>` | 产物 meta.json 全文；`DELETE ?version=vN` 删除该版本（危险） |
| GET | `/datasets` | 数据集体检（内置 .mat + `data/datasets` 下上传的表格数据集） |
| GET | `/datasets/db` | Datasets 表登记记录；`POST` 登记新数据集 |
| POST | `/datasets/upload` | 上传表格文件到 `data/datasets/<名称>/`（multipart：`name` + 多个 `file`） |
| GET | `/datasets/table` | 表格预览（列统计 / 前 N 行 / 推荐信号列，`?path=`） |
| GET | `/datasets/signal` | 取一段原始信号（.mat 或表格，画波形用） |
| POST | `/train` | 训练 → 落盘产物 → 写 `Trainings` |
| GET | `/trainings` | 最近训练记录（读库） |
| POST | `/predict` | 推理 → 写 `InferenceTasks` + `InferenceResults` + `ModelInvocations` |
| GET | `/inference-tasks` | 最近推理任务（读库） |
| GET | `/inference-tasks/<id>` | 任务 + 结果明细（读库） |
| GET | `/system` | 运行信息（Python/依赖版本/路径/库表行数/占用） |
| GET | `/system/logs` | 训练日志列表；`/system/logs/<name>?tail=N` 看尾部 |
| POST | `/system/maintenance` | 维护（目前支持 `target=figures` 清空图库，危险） |
| GET | `/figures` | 已生成的图（训练曲线/混淆矩阵/每类指标/预测分布/预测波形） |
| GET | `/figures/<路径>` | 直接返回 PNG（浏览器打开即可看） |
| — | `/todos` 等 | 原有示例接口，**未改动** |

```bash
# 训练（1dcnn / cwt_cnn / adtk，也可写 算法模型1/2/3）
curl -X POST http://127.0.0.1:5000/train -H "Content-Type: application/json" \
     -d '{"model":"1dcnn","epochs":10}'

# 推理：既可以用文件（工作区内的 .mat/.csv/.npy，按窗口切），也可以直接送数组
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
     -d '{"model":"1dcnn","path":"1DCNN/0HP/normal_0_97.mat","index":0,"limit":3}'
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
     -d '{"model":"1dcnn","samples":[[0.1,0.2,...共784个数]]}'
```

`POST /train` 常用参数：`epochs` `batch_size` `number` `length` `stride` `rate` `seed`
`strict`（是否跳过越界窗口）`legacy_scaler`（是否复刻旧脚本的标准化口径）；
adtk 另有 `detector` `k` `baseline_file` `max_points` `factor`。

## 二、产物约定（流程图里的「Pxl模型」）

```
data/models/<模型名>/v1/model.keras | model.h5 | model.pt | detector.pkl
                       v1/scaler.npz     ← 训练期标准化参数（推理必须复用）
                       v1/meta.json      ← 输入长度、类别表、指标、超参、数据集指纹
data/logs/train-<模型>-<时间>.log          ← 训练全过程日志
data/model_management.db                   ← 未接真库时的 SQLite 兜底库
```

`meta.json` 里**必须**带类别表（`labels`）：否则模型文件本身无法解释 0..9 对应哪种故障。

## 二·补、出图（matplotlib，落 PNG 不弹窗）

原来"表现"结果的地方只有三个脚本里的 `plt.show()`（1DCNN 的准确率/损失曲线、cwt_cnn 的混淆矩阵、
adtk 的时序图），**进程一退图就没了，也拿不进接口**。现在统一由 `model_service/figures.py`
用 matplotlib 的 **Agg 后端 + `savefig`** 落盘：

```
data/figures/<模型>/<版本>/training_curves.png        准确率/损失（训练集 vs 验证集）
                        /confusion_matrix.png         混淆矩阵（带计数标注）
                        /per_class_metrics.png        每类 精确率/召回率/F1
                        /predict-<时间戳>/prediction_distribution.png   本次推理的预测分布
                                        /predicted_windows.png        窗口原始信号 + 预测标签
```

- 训练完成自动出 3 张，每次 `/predict` 自动出 2 张；响应里带 `figures[]`（含可直接打开的 `url`）
  与 `figures_dir`；推理图目录同时写进 `InferenceTasks.OutputPath`，训练图目录写进 `Trainings.Remark`。
- **出图失败不影响训练/推理**：只在响应里回一个 `figures_error`。
- 中文字体按 `Microsoft YaHei → SimHei → SimSun` 顺序自动挑选（本机三个都在），
  负号与刻度正常；`MPLCONFIGDIR` 被指到 `data/.cache/matplotlib`，避免受限环境写用户目录失败。
- 彩蛋级的坑：`✓/✗` 这类符号在雅黑/宋体里**没有字形**，会渲染成方框，所以命中/未命中用中文标。

## 二·补、表格数据集（Excel / CSV）

除了 CWRU 的 `.mat`，服务还支持**表格格式的数据集**，约定是：
**一个文件 = 一个类别，文件名即标签，表内指定一列作为振动信号。**

- 支持 `.csv` `.txt` `.xlsx` `.xls`（xlsx 走 openpyxl，xls 走 xlrd；CSV 依次试 utf-8-sig / utf-8 / **gbk**，
  中文 Excel 导出的 CSV 常见 GBK）
- 上传：控制台「数据集管理 → 表格数据集」，或 `POST /datasets/upload`（multipart，字段 `name` + 多个 `file`），
  落到 `data/datasets/<数据集名>/`
- 信号列选择：显式 `column=`/`signal_column=` 优先；否则按「列名像信号」（振幅/振动/加速度/value/signal/…）
  → 「首个有波动的数值列」的顺序自动挑，**时间/序号列会被排除**。推理时也能用 `column` 覆盖。
- 训练：`POST /train` 传 `dataset_type=tabular` + `dataset_dir` +（可选）`signal_column`，
  其余 `length/number/stride/rate/seed/strict` 与 `.mat` 完全一致——两个数据源共用同一套切窗、
  标准化、划分逻辑（`datasets.finalize_windows`），所以 1DCNN / cwt_cnn 一行没改。
- 越界窗口的处理与 `.mat` 一致：`strict=true` 跳过并在结果里回报，**绝不补 NaN**。
- 实测（`data/datasets/DEMO-轴承表格数据`：4 个文件 = 4 类，3 个 CSV + 1 个 xlsx，
  每文件 40000 行 × 3 列，故意混入「时间」「温度」两列干扰）：自动识别出信号列 `振动幅值`；
  `length=512, number=280, stride=128, epochs=30` → 测试准确率 **1.0000**、验证 0.9940，
  推理 CSV/xlsx 均给出 ~0.999 的置信度。（这是我自己造的演示数据，类别区分度大，重点在链路通。）

## 三、数据库

```
① Datasets ─┐
            ├─▶ ② Models ─▶ ③ Trainings ─┬─▶ ④ ModelInvocations
② Models  ──┘                            └─▶ ⑤ InferenceTasks ─▶ ⑥ InferenceResults
                                               （另需 TargetDatasetID→①、ModelID→②）
```

- **权威锚点**：`InferenceTasks.TrainingID`（不是 DeploymentID）。理由是推理结果的可信度取决于
  「哪一次训练」，部署只是同一次训练的投放位置；因此 `DeploymentID`/`DeviceID` 留空，
  等「边缘设备」支线落地后再回填。这条正是 `sql/schema.sql` 末尾自己标注的悬案。
- **`TargetDatasetID` 是 NOT NULL**：内联样本没有"数据集"概念，服务会登记一条
  `ADHOC-<模型名>` 数据集，而不是为了满足外键去伪造真实数据集。
- 三种方言由环境变量切换，表结构分别复用仓库里已有的脚本：

| `MODEL_DB_DIALECT` | 驱动 | 建表脚本 |
|---|---|---|
| `mysql` | pymysql | `sql/schema_mysql.sql`（**本机当前使用**） |
| `sqlite` | 标准库 | `sql/schema_sqlite.sql`（本次新增的镜像表，零配置兜底） |
| `sqlserver` | pyodbc | `sql/schema.sql`（按 `GO` 分批，库不存在会先在 master 里建） |

其余变量：`MODEL_DB_HOST` `MODEL_DB_PORT` `MODEL_DB_USER` `MODEL_DB_PASSWORD`
`MODEL_DB_NAME` `MODEL_DB_ODBC_DRIVER` `MODEL_DB_TRUSTED`。

**本机现状（已接通）**：MySQL 9.2 @ `127.0.0.1:3306`，库 `model_management`，
应用账号 `ljx666`（已授权 `model_management.*`）。这些值写在项目根的 **`db.env`** 里，
服务启动时自动读取，所以直接 `python main.py` 即可，不需要 export 任何变量；
`db.env.example` 是不含口令的同款模板，正式环境请自行替换账号或改回 `root`。

> 注意：`db.env` 含明文口令，不要外发或提交版本库。
>
> SQL Server 那一路本机**跑不通**：`MSSQL$SQLEXPRESS` 服务虽然在跑，但机器上只有过时的
> `SQL Server` / `SQL Server Native Client 10.0` ODBC 驱动，用信任连接报"安全包中没有可用的凭证"、
> 用旧驱动报 SSL 错误；要用它得先装微软官方 ODBC Driver 17/18。

## 四、与既有脚本的关系（改了什么、没改什么）

| 文件 | 处理 |
|---|---|
| `1DCNN/1DCNN.py`、`1DCNN/preprocessing.py` | **未改动**，命令行行为完全一致 |
| `cwt_cnn/preprocess.py` | **未改动** |
| `cwt_cnn/cwt_cnn_pytorch.py` | 重构：训练主体收进函数、入口移入 `__main__`，`python cwt_cnn_pytorch.py` 表现不变；`fc1` 输入维度由硬编码 `16*196` 改为按 `length` 推导 |
| `main.py` | 只加了 `register_api(api)` 与 `threaded=True` |
| `sql/schema*.sql` | 未改动，新增一份 `schema_sqlite.sql` |

服务侧**没有**复用 `1DCNN/preprocessing.py` 的切片逻辑，而是新增 `datasets.py`，差异都是刻意的：

1. **跳过越界窗口而不是补 NaN**（`strict=True`）：原逻辑对长度不足的文件会切出短数组再补成整行
   NaN，实测 0HP 里 `IR014` 只有 63788 点，导致验证/测试集各有 10% 的全 NaN 样本，
   NaN 子集准确率恒为 0。服务侧改为跳过并在响应/库备注里回报跳过数量。
2. **标准化只用训练集拟合**（原脚本把训练+测试拼起来拟合，属于统计量泄漏）。
3. **固定随机种子**（原 `StratifiedShuffleSplit` 未固定，同参数两次跑出过 0.5933 与 0.750）。
4. **类别号写死**：原 `add_labels()` 的类别号来自 `os.listdir()` 顺序，文件名一改就错位；
   现在用显式映射表，并随产物存进 `meta.json`。
5. **标准化参数随模型落盘**，推理侧套用同一套均值/方差——第一次端到端验证时正是漏了这步，
   测试集 0.83 的模型对原始信号窗口的预测全是错的。

## 五、明确的已知限制

1. **adtk 分支判别力不足**（按你的要求已搁置，代码保留但未训练产物）：以正常轴承信号为基线
   fit `PcaAD(k=1)`，在原始振动窗口上，故障窗口的异常点占比（0.026~0.103）反而**低于**基线参考
   占比（0.137），标定后所有窗口都判为"正常"。要做成有用的异常检测，得先做特征工程
   （CWT 时频图/统计特征），这也正是流程图里「算法模型3」还缺的一块。
2. **`/train` 是同步阻塞的**，没有任务队列；开发服务器开了 `threaded=True`，
   但一条训练请求会占住一个线程（实测 1DCNN 10 epoch 约 15 秒，cwt_cnn 50 epoch 约 40 秒）。
3. **`model_service` 不写模型版本号**（只有自增的 `v1/v2`），`ModelDeployments` 表和
   「边缘设备 → 机床」那条支线仍未接入，`EdgeDevices` 表还是空的。
4. **环境相关的坑**（已规避，换机器可能不再复现）：
   - 某些受限环境禁止在 `mkdtemp` 建的目录里写文件 → Keras 原生 `.keras`（zip，先写临时文件再改名）
     必然失败，代码会自动回退到 h5py 直写的 `.h5`；失败的 `.keras` 半成品会被显式删除，
     否则它（只有 config.json、没有权重）会被权重查找误命中。
   - pip 在这类环境下也装不了包（同样的临时目录限制），本次 `pymysql` / `pyodbc`
     是直接解包 wheel 到 site-packages 安装的。
