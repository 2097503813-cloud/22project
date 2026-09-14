# 临时检查脚本（跑完即删）：列出 model_service 下所有函数/方法/属性，标出没有 docstring 的
import ast
import pathlib

for path in sorted(pathlib.Path('model_service').glob('*.py')):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    missing = []
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total += 1
            if not ast.get_docstring(node):
                missing.append((node.lineno, type(node).__name__.replace('Def', ''), node.name))
    if missing:
        print(f'\n{path}  （{len(missing)}/{total} 个定义没有 docstring）')
        for lineno, kind, name in sorted(missing):
            print(f'  {lineno:4d}  {kind:10s} {name}')
    else:
        print(f'\n{path}  ✅ {total} 个定义全部有 docstring')
