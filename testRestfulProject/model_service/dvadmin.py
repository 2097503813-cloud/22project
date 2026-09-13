# -*- coding: utf-8 -*-
"""Django-Vue3-Admin（dvadmin）兼容层 —— 让 frontend/22project 直接跑在本服务上。

frontend/22project 是 dvadmin 的前端工程，它启动时会打一组 Django 接口；这里用 Flask
把**最小必需集合**实现出来，数据全部来自本服务（模型产物 / 数据集 / 图 / 库表）：

    POST /api/login/                              登录   → {code:2000, data:{access, username}}
    GET  /api/system/user/user_info/               用户信息
    GET  /api/system/menu/web_router/              动态菜单（后端控制路由，五个模块在这里）
    GET  /api/init/dictionary/?dictionary_key=all  字典（data 必须是数组）
    GET  /api/system/message_center/get_newest_msg/ 站内消息（空数组即可）
    GET  /api/captcha/                             验证码（返回 captcha_state=false，登录页不显示）
    POST /api/logout/                              退出

说明：
  * 前端 axios 的约定是 `{code, data, msg}`，code=2000 表示成功；
    而 model_service 自己的业务接口是裸 JSON —— 两者在同一端口上共存，互不影响
    （裸 JSON 没有 code 字段，前端的拦截器会直接放行）。
  * 登录只做形式校验（任意非空账号口令都通过）：这是一个局域网内的演示平台，
    没有用户体系；真要接权限，把这里换成真实的鉴权即可。
  * 同时装上 CORS 头：前端 dev server 在 8080、本服务在 5000，属于跨域。
"""

from __future__ import annotations

import uuid

from flask import Blueprint, jsonify, request

# 五个模块（后端控制路由：前端的 dynamicRoutes[0].children 会被这份数据替换）
def _menu_payload() -> list[dict]:
    # 注意：这里**不要**再放"首页"。模板里 stores/frontendMenu.ts 已经写死了一个 /home
    # （已改为指向本平台的 platform/home/index），两边都下发就会出现"两个首页"。
    # 路由守卫登录成功后 next('/home')、dynamicRoutes[0].redirect 也是 /home，与它一致。
    modules = [
        (102, "模型管理", "ele-Cpu", "/platform/model", "platformModel", "platform/model/index", False),
        (103, "数据集管理", "ele-Coin", "/platform/dataset", "platformDataset", "platform/dataset/index", False),
        (104, "数据展示", "ele-DataLine", "/platform/visual", "platformVisual", "platform/visual/index", False),
        (105, "系统管理", "ele-Setting", "/platform/system", "platformSystem", "platform/system/index", False),
    ]
    return [{
        "id": mid, "parent": None, "title": title, "icon": icon,
        "web_path": path, "component_name": comp_name, "component": component,
        # cache=False → 不加入 KeepAlive：每次切回该页都重新挂载，onMounted 里的
        # 数据加载会重新执行。之前给 True，页面被缓存后切回来不重新挂载，
        # 一旦首次加载失败（或后端重启过）就会一直显示空，必须先刷新浏览器。
        "visible": True, "cache": False, "is_affix": affix, "is_iframe": False,
        "is_catalog": False, "is_link": False, "link_url": "",
    } for mid, title, icon, path, comp_name, component, affix in modules]


def _ok(data=None, msg: str = "success"):
    return jsonify({"code": 2000, "data": data, "msg": msg})


def _user_payload() -> dict:
    return {
        "id": 1, "username": "admin", "name": "管理员", "avatar": "",
        "email": "admin@localhost", "mobile": "", "gender": "1",
        "dept_info": {"dept_id": 1, "dept_name": "轴承故障诊断平台"},
        "role_info": [{"id": 1, "name": "超级管理员", "key": "admin"}],
        "roles": ["admin"], "is_superuser": True, "pwd_change_count": 1,
        "description": "本地演示账号（由 model_service 兼容层提供）",
    }


def build_blueprint() -> Blueprint:
    bp = Blueprint("dvadmin", __name__)

    @bp.post("/api/login/")
    def login():
        body = request.get_json(silent=True) or {}
        username = (body.get("username") or "").strip()
        if not username:
            return jsonify({"code": 4000, "data": None, "msg": "用户名不能为空"}), 200
        # 注意：登录响应必须带上 pwd_change_count，且要比 0 大。
        # 前端的登录成功分支是：
        #   if (data.pwd_change_count == 0) return router.push('/login');   // 强制改密码
        #   ... loginSuccess() 里 if (pwd_change_count > 0) 才会 router.push('/home')
        # 少了这个字段（undefined）两边都进不去，表现就是"登录后页面不跳转"。
        data = _user_payload()
        data.update({"access": uuid.uuid4().hex, "refresh": uuid.uuid4().hex,
                     "username": username, "pwd_change_count": 1})
        return _ok(data, "登录成功")

    @bp.post("/api/logout/")
    def logout():
        return _ok(None, "已退出")

    @bp.get("/api/system/user/user_info/")
    def user_info():
        return _ok(_user_payload())

    @bp.post("/api/system/user/update_user_info/")
    def update_user_info():
        data = dict(_user_payload())
        data.update(request.get_json(silent=True) or {})
        return _ok(data, "已更新（本地演示不会真的落库）")

    @bp.post("/api/system/user/change_password/")
    @bp.post("/api/system/user/login_change_password/")
    def change_password():
        return _ok(None, "本地演示环境不需要改密码")

    @bp.get("/api/system/menu/web_router/")
    def web_router():
        return _ok(_menu_payload())

    @bp.get("/sse/")
    def sse_stub():
        """dvadmin 的站内消息推送（前端用 EventSource 连 /sse/?token=...）。

        本地演示环境不做推送：返回一个合法的空事件流，并把重连间隔设成 1 小时，
        免得前端不停重连、控制台一直刷 404 与「连接已关闭」。
        """
        return "retry: 3600000\n\n", 200, {"Content-Type": "text/event-stream",
                                           "Cache-Control": "no-cache"}

    @bp.get("/api/init/dictionary/")
    def dictionary():
        return _ok([], "本地演示环境暂无字典数据")

    @bp.get("/api/init/settings/")
    def settings():
        # 前端把这个对象直接当字典读，例如 systemConfig['base.captcha_state']
        return _ok({"base.captcha_state": False, "base.site_name": "轴承故障诊断平台",
                    "base.login_title": "轴承故障诊断平台"})

    @bp.get("/api/system/menu_button/menu_button_all_permission/")
    def menu_button_all_permission():
        # 按钮权限清单：前端拿到后逐条 forEach，必须是数组
        return _ok([], "本地演示环境不做按钮级权限")

    @bp.get("/api/system/message_center/get_newest_msg/")
    @bp.get("/api/system/message_center/get_self_receive/")
    def message_center():
        return _ok([], "无消息")

    @bp.get("/api/captcha/")
    def captcha():
        # None → 登录页的 isShowCaptcha 为假，不显示验证码框
        return _ok({"captcha_state": False, "key": None, "image_base64": None})

    @bp.get("/api/system/system_config/get_table_data/")
    def system_config():
        return _ok({"base.captcha_state": False, "base.site_name": "轴承故障诊断平台"})

    @bp.get("/api/system/dept/all_dept/")
    @bp.get("/api/system/dept/dept_all/")
    def all_dept():
        # 前端用 XEUtils.toArrayTree(ret.data, {parentKey:'parent'}) 建树：
        # data 必须是**数组**，且每项要有 id / parent（之前返回 {results,total} 会报
        # "Cannot create property 'id' on number '0'"）
        return _ok([{"id": 1, "parent": None, "name": "轴承故障诊断平台",
                     "dept_name": "轴承故障诊断平台", "key": 1, "owner": [], "status": True}])

    @bp.get("/api/dvadmin3_social_oauth2/backend/get_login_backend/")
    def login_backend():
        return _ok([], "未启用第三方登录")

    @bp.get("/api/system/role/")
    @bp.get("/api/system/user/")
    @bp.get("/api/system/area/")
    def fast_crud_list():
        # fast-crud 的分页列表约定：{results, total}
        return _ok({"results": [], "total": 0}, "本地演示环境暂无数据")

    return bp


def register_dvadmin(app) -> None:
    """注册兼容接口，并给**整个应用**装上 CORS（前端 8080 → 本服务 5000 是跨域）。"""
    app.register_blueprint(build_blueprint())

    @app.after_request
    def _cors(response):                       # noqa: ANN001
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "Access-Control-Request-Headers", "Content-Type,Authorization")
        response.headers["Access-Control-Max-Age"] = "86400"
        return response

    @app.errorhandler(404)
    def _not_found(err):                       # noqa: ANN001
        """未实现的 /api/** 一律回 JSON（而不是 405/404 的 HTML 页面），方便定位缺哪个接口。

        注意：这里**不能**用 `@app.route('/api/<path:...>', methods=['OPTIONS'])` 那种兜底路由——
        它会参与 URL 匹配，把未注册的 /api/xxx 请求截成 405 METHOD NOT ALLOWED。
        CORS 预检由 Flask 对已注册路由自动处理，再经上面的 after_request 补头即可。
        """
        if request.path.startswith("/api/"):
            return jsonify({
                "code": 404,
                "data": None,
                "msg": f"model_service 的 dvadmin 兼容层尚未实现该接口：{request.path}",
            }), 404
        return err

