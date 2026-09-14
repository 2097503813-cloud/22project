"""model_service —— 让「训练 → 模型产物 → 推理」在 Web 接口上真实跑通的最小服务层。

模块划分（对应流程图上的盒子）：
    config.py     运行期配置（路径 + 数据库连接，全部可用环境变量覆盖）
    datasets.py   数据集与类别标签的**显式**映射，以及滑窗取样
    registry.py   模型产物落盘 / 加载（流程图里的「Pxl模型」盒子）
    db.py         数据库写入层，按外键顺序写 Trainings → InferenceTasks → InferenceResults
    training.py   训练编排（算法模型1=1DCNN / 算法模型2=cwt_cnn / 算法模型3=adtk）
    inference.py  推理（流程图里的「推理」盒子）
    api.py        flask_restful 接口资源（流程图里的「Web访问」盒子）
"""
from .config import config  # noqa: F401
__all__ = ["config"]
