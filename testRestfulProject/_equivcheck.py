# 临时检查脚本（跑完即删）：剥离注释与 docstring 后，对比 HEAD 与工作区的 .py 代码是否一致。
import ast, io, subprocess, sys

FILES = ["model_service/__init__.py","model_service/api.py","model_service/config.py",
    "model_service/datasets.py","model_service/db.py","model_service/dvadmin.py",
    "model_service/figures.py","model_service/inference.py","model_service/registry.py",
    "model_service/tabular.py","model_service/training.py",
    "main.py","createdoc.py","_selftest_upload.py"]

def norm(source, name):
    """解析成 AST，把 docstring 全部去掉后 dump —— 只剩"真正的代码结构"。"""
    tree = ast.parse(source, filename=name)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
               and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)

bad = 0
for rel in FILES:
    head = subprocess.run(["git","show",f"HEAD:testRestfulProject/{rel}"],
                          capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    now = io.open(rel, encoding="utf-8").read()
    try:
        a, b = norm(head, rel), norm(now, rel)
    except SyntaxError as exc:
        print(f"✗ {rel}: 语法错误 {exc}"); bad += 1; continue
    if a == b:
        print(f"✅ {rel}: 去掉注释/docstring 后 AST 完全一致")
    else:
        bad += 1
        print(f"❌ {rel}: AST 不同（长度 {len(a)} vs {len(b)}）")
        i = next((k for k in range(min(len(a),len(b))) if a[k]!=b[k]), min(len(a),len(b)))
        print(f"     首个差异 @ {i}")
        print(f"     HEAD: {a[max(0,i-80):i+80]}")
        print(f"     当前: {b[max(0,i-80):i+80]}")
print("\n" + ("全部一致 ✅" if not bad else f"{bad} 个文件有代码差异 ❌"))
sys.exit(1 if bad else 0)
