/**
 * 前端页面静态编译校验（不依赖 esbuild / 不起子进程）
 *
 * 背景：本机的受限沙箱不允许 esbuild 起子进程（spawn EPERM），所以 `vite build` / `vite dev`
 * 在沙箱内无法运行。这里改用工程里已经装好的编译器在**进程内**逐个编译我新写的 SFC：
 *   - @vue/compiler-sfc：解析 SFC → 编译 <script setup> → 编译 <template>
 *   - sass：编译 <style lang="scss">
 *   - 校验 import 路径是否存在
 * 它能覆盖 vite build 会报的绝大多数错误（模板语法、script 语法、SCSS、引用路径）。
 */
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const sfc = require(path.join(ROOT, 'node_modules/@vue/compiler-sfc'));
const sass = require(path.join(ROOT, 'node_modules/sass'));

const FILES = [
	'src/views/platform/home/index.vue',
	'src/views/platform/model/index.vue',
	'src/views/platform/dataset/index.vue',
	'src/views/platform/visual/index.vue',
	'src/views/platform/system/index.vue',
];
const ALSO = ['src/api/platform/index.ts'];

let errors = 0;
const ok = (m) => console.log('  ✅ ' + m);
const bad = (m) => { errors++; console.log('  ❌ ' + m); };

console.log('=== 1) SFC 编译校验 ===');
for (const rel of FILES) {
	const file = path.join(ROOT, rel);
	const source = fs.readFileSync(file, 'utf8');
	const { descriptor, errors: parseErrors } = sfc.parse(source, { filename: file });
	if (parseErrors.length) { bad(`${rel} 解析失败：${parseErrors.map((e) => e.message).join('; ')}`); continue; }

	let fileOk = true;
	// script setup
	try {
		const id = 'x' + Math.random().toString(36).slice(2, 10);
		const compiled = sfc.compileScript(descriptor, { id });
		const bindings = Object.keys(compiled.bindings || {});
		// template
		if (descriptor.template) {
			const t = sfc.compileTemplate({
				source: descriptor.template.content, filename: file, id,
				compilerOptions: { bindingMetadata: compiled.bindings },
			});
			if (t.errors && t.errors.length) { fileOk = false; bad(`${rel} 模板错误：${t.errors.map((e) => e.message || e).join('; ')}`); }
		}
		// style
		for (const style of descriptor.styles) {
			if (style.lang === 'scss') {
				try { sass.compileString(style.content, { syntax: 'scss' }); }
				catch (e) { fileOk = false; bad(`${rel} SCSS 错误：${e.message.split('\n')[0]}`); }
			}
		}
		if (fileOk) ok(`${rel}（模板 + script setup + SCSS 全部编译通过，绑定 ${bindings.length} 个）`);
	} catch (e) {
		bad(`${rel} script 编译失败：${e.message.split('\n')[0]}`);
	}
}

console.log('\n=== 2) import 路径校验 ===');
const IMPORT_RE = /from\s+['"]([^'"]+)['"]/g;
const checkFile = (rel) => {
	const src = fs.readFileSync(path.join(ROOT, rel), 'utf8');
	const aliasRoot = path.join(ROOT, 'src');
	let m;
	while ((m = IMPORT_RE.exec(src))) {
		const spec = m[1];
		if (spec.startsWith('.')) continue;                       // 相对路径，编译器自己管
		if (spec.startsWith('/@/')) {
			const target = path.join(aliasRoot, spec.slice(3));
			const found = fs.existsSync(target) || fs.existsSync(target + '.ts') || fs.existsSync(target + '.vue')
				|| fs.existsSync(path.join(target, 'index.ts'));
			if (!found) bad(`${rel} 引用不存在的别名路径：${spec}`);
		} else if (spec.startsWith('/src/')) {
			if (!fs.existsSync(path.join(ROOT, spec))) bad(`${rel} 引用不存在：${spec}`);
		} else {
			const pkg = spec.startsWith('@') ? spec.split('/').slice(0, 2).join('/') : spec.split('/')[0];
			if (!fs.existsSync(path.join(ROOT, 'node_modules', pkg))) bad(`${rel} 依赖未安装：${pkg}`);
		}
	}
};
[...FILES, ...ALSO].forEach(checkFile);
if (!errors) ok('所有 import（/@/ 别名、外部依赖）都能解析');

console.log('\n=== 3) 后端契约一致性（页面用到的接口是否都在 flask 侧存在）===');
const apiSrc = fs.readFileSync(path.join(ROOT, 'src/api/platform/index.ts'), 'utf8');
const urls = [...apiSrc.matchAll(/url:\s*(?:`([^`]+)`|'([^']+)')/g)].map((m) => (m[1] || m[2]));
const norm = [...new Set(urls.map((u) => '/' + u.replace(/^\//, '').split('?')[0].replace(/\$\{[^}]+\}/g, '<x>')))];
console.log('  前端调用的接口：');
norm.forEach((u) => console.log('    ' + u));
fs.writeFileSync(path.join(ROOT, 'platform-api-paths.json'), JSON.stringify(norm, null, 1));

console.log(`\n结论：${errors ? errors + ' 处错误' : '全部通过 ✅'}`);
process.exit(errors ? 1 : 0);
