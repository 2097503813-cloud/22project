# 轴承故障诊断模型管理平台

把「CWRU 轴承振动数据 → 训练模型 → 推理 → 落库 → 前端展示」串成一条可跑通的流水线：
Flask 后端（`testRestfulProject/`）+ Vue3 前端（`frontend/22project/`，基于 django-vue3-admin）+ MySQL。

> **先看文档，再看代码**：
> - 📘 [`项目交接文档.md`](项目交接文档.md) —— 怎么跑起来、接口清单、数据库表、踩坑记录、遗留问题（**接手第一份**）
> - 📗 [`项目技术详解.md`](项目技术详解.md) —— 数据集格式、训练/推理/记录/任务四个模块的逐步拆解、Bug 清单
> - 📙 [`testRestfulProject/model_service/README.md`](testRestfulProject/model_service/README.md) —— 后端业务包的接口/产物约定/与原始脚本的关系

---

## 目录结构

```
D:\22project\
├─ testRestfulProject\              ← 后端（Flask + flask_restful）
│  ├─ main.py                       唯一入口：注册 /todos 之外的业务接口 + dvadmin 兼容层
│  ├─ model_service\                业务包（下面有模块地图）
│  ├─ 1DCNN\0HP\*.mat               ★ 数据集①：CWRU 48k 驱动端，10 个文件 = 10 类
│  ├─ data\
│  │  ├─ models\<模型名>\vN\        ★ 模型产物：model.h5 / model.pt / detector.pkl + scaler.npz + meta.json
│  │  ├─ datasets\<数据集名>\       ★ 数据集②：上传的表格数据集（csv/xlsx，一文件一类别）
│  │  ├─ figures\                   训练/推理自动出的图（PNG）
│  │  ├─ logs\                      每次训练的 stdout 日志
│  │  └─ tmp\                       临时目录（受限环境下会退化成这里）
│  ├─ adtk\                         第三方时序异常检测库（整份源码拷贝，离线可用）
│  ├─ sql\schema_mysql.sql          建库脚本（8 张表）
│  ├─ db.env                        数据库连接配置（git 外）
│  └─ _selftest_upload.py           后端自测：模型探测 / 上传 / 改名
├─ frontend\22project\              ← 前端（Vue3 + Vite + Element Plus + Pinia）
│  └─ src\
│     ├─ views\platform\{home,model,dataset,visual,system}\   五个业务模块
│     ├─ api\platform\index.ts      平台接口封装
│     └─ utils\platformRequest.ts   专用 axios 实例（绕开框架的信封校验）
├─ 项目交接文档.md
└─ 项目技术详解.md
```

### 后端模块地图（`testRestfulProject/model_service/`）

| 模块 | 职责 |
|---|---|
| `config.py` | 路径/环境/默认超参（`defaults` 里三个模型的默认值） |
| `datasets.py` | **数据集①** CWRU `.mat`：读 DE 通道、定类别、切窗、数据集体检 |
| `tabular.py` | **数据集②** 表格：读 csv/xlsx、挑信号列、切窗、目录体检 |
| `training.py` | 训练编排：`train()` 统一入口 + 三个 trainer（1dcnn / cwt_cnn / adtk） |
| `inference.py` | 推理：校验 → 切窗 → scaler → 分框架预测 → 组装结果 |
| `registry.py` | 产物版本管理：`data/models/<名>/vN/` 的读写删 |
| `db.py` | MySQL 访问层：8 张表的 CRUD、外键顺序、连接复用、行数缓存 |
| `figures.py` | 出图（无头 matplotlib，Agg 后端） |
| `api.py` | flask_restful 资源（业务接口，返回**裸 JSON**） |
| `dvadmin.py` | django-vue3-admin 所需的最小兼容接口（返回 `{code,data,msg}` 信封）+ CORS |
| `console.html` | 零构建调试控制台（`/ui`），前端覆盖后保留作备用入口 |

---

## 怎么跑起来

### 1. 数据库（MySQL，本机 3306）

```sql
-- 库：model_management；账号：ljx666 / 123456
source D:\22project\testRestfulProject\sql\schema_mysql.sql;
```
连接配置在 `testRestfulProject/db.env`（`MODEL_DB_DIALECT=mysql`、HOST/PORT/USER/PASSWORD/NAME）。
没有真库时 `model_service` 会自动退化成 SQLite，接口照样能跑通。

### 2. 后端（**必须先起**）

```powershell
cd D:\22project\testRestfulProject
venv\Scripts\python.exe main.py            # → http://127.0.0.1:5000
```

### 3. 前端

```powershell
cd D:\22project\frontend\22project
npm run dev                                # → http://127.0.0.1:8080
```

### 三个地址

| 地址 | 说明 |
|---|---|
| http://127.0.0.1:5000/api | 接口索引（JSON） |
| http://127.0.0.1:5000/ui | 零构建控制台（备用入口） |
| http://127.0.0.1:8080 | 平台前端（首页 / 模型管理 / 数据集管理 / 数据展示 / 系统管理） |

---

## 三个模型

| 模型 | 类型 | 框架 | 数据源 | 产物 |
|---|---|---|---|---|
| `1DCNN` | 分类（10 类轴承状态） | TensorFlow / Keras | .mat 或表格 | `model.h5` + `scaler.npz` |
| `cwt_cnn` | 分类（同任务，PyTorch 实现） | PyTorch | .mat 或表格 | `model.pt` + `scaler.npz` |
| `adtk` | **无监督异常检测** | adtk（PcaAD） | 只要一段"正常"基线 | `detector.pkl` |

类别编号与中文标签的固定映射见 `datasets.py` 的 `CWRU_0HP_CLASSES`（10 类，顺序即 class_id）。

## 两套数据集

| 数据集 | 位置 | 格式 | 约定 |
|---|---|---|---|
| CWRU（内置只读） | `testRestfulProject/1DCNN/0HP/*.mat` | MATLAB `.mat`，取 DE 通道 | **一个文件 = 一个类别**，10 个文件 10 类 |
| 表格（可上传） | `testRestfulProject/data/datasets/<数据集名>/` | `.csv/.txt/.xlsx/.xls` | **一个文件 = 一个类别，文件名即标签**，表内自选一列作信号 |

---

## 常用命令

```powershell
# 训练（1DCNN，10 轮）
curl -X POST http://127.0.0.1:5000/train -H "Content-Type: application/json" -d "{\"model\":\"1dcnn\",\"epochs\":10}"
# 训练 adtk（无监督：窗口当样本 + PCA 重构误差）
curl -X POST http://127.0.0.1:5000/train -H "Content-Type: application/json" -d "{\"model\":\"adtk\",\"k\":4}"
# 推理
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"model\":\"1dcnn\",\"path\":\"1DCNN/0HP/normal_0_97.mat\",\"index\":0,\"limit\":3}"
# 模型档案（登记信息 + 参数 + 指标 + 类别表 + 最近训练 + 引用统计）
curl http://127.0.0.1:5000/models/1dcnn/overview

# 前端页面编译校验（不需要 esbuild / dev server）
node D:\22project\frontend\22project\check-platform.cjs
# 后端自测（模型探测 / 上传 / 改名，跑完自动清理）
cd D:\22project\testRestfulProject; venv\Scripts\python.exe _selftest_upload.py
```

---

## 已知限制

- **沙箱环境**：本机 DSH 工作区沙箱起不来（`windows-acl-run: --temp is not an existing directory`），涉及写临时目录 / 管道子进程的操作（pip 安装、`vite build`）会被拒；前端因此用 `check-platform.cjs` 做等价编译校验。
- **训练是同步阻塞的**：`/train` 一次请求跑完才返回（1DCNN 10 轮约 15s，cwt_cnn 50 轮约 25s），不要并发压。
- **`IR014` 数据点不足**：`48k_Drive_End_IR014_0_174.mat` 只有 63,788 点，凑不满 784×窗数 时**跳过越界窗口、不做 NaN 补齐**，所以该类测试集为空、准确率里它恒为 0。
- **adtk 精度有限**：CWRU 上能做到正常文件 0/20 误报、故障文件 19~20/20 命中，但它只回答"是否异常"，不分类别。
