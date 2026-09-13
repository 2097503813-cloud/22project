#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""后端唯一入口：把业务接口 + dvadmin 兼容层挂到一个 Flask app 上。

启动：`venv\\Scripts\\python.exe main.py`  → http://127.0.0.1:5000

两层路由：
  model_service.api     业务接口（返回**裸 JSON**）：/api /health /models /datasets /train /trainings /predict …
  model_service.dvadmin 前端 django-vue3-admin 要的兼容接口（返回 `{code,data,msg}` 信封）+ CORS

历史上这里还有一套 flask_restful 官方示例的 `/todos` 资源与 `PcaAD` 演示代码，
与本平台无关，已删除；需要看原示例请回查 git 历史或 flask_restful 文档。
"""
import sys
from pathlib import Path

from flask import Flask
from flask_restful import Api

# 让 `adtk/`、`1DCNN/` 这些原项目目录可以被 import（trainer 里按需插入，这里兜个底）
sys.path.append(str(Path(__file__).parent))

from model_service.api import register_api                    # noqa: E402
from model_service.dvadmin import register_dvadmin            # noqa: E402

app = Flask(__name__)
api = Api(app)

register_dvadmin(app)          # 先注册兼容层（含 404 兜底与 CORS，必须在业务接口之前）
register_api(api)              # 再注册业务接口

if __name__ == '__main__':
    # threaded=True：/train 是同步阻塞的，避免一条训练请求把整个服务卡住
    app.run(debug=True, threaded=True)
