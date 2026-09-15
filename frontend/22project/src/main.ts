import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import { directive } from '/@/directive/index';
import { i18n } from '/@/i18n';
import other from '/@/utils/other';
import '/@/assets/style/tailwind.css'; // 先引入tailwind css, 以免element-plus冲突
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
import '/@/theme/index.scss';
import mitt from 'mitt';
import VueGridLayout from 'vue-grid-layout';
import piniaPersist from 'pinia-plugin-persist';
// @ts-ignore
import fastCrud from './settings.ts';
import pinia from './stores';
import { Local, Session } from '/@/utils/storage';
import {RegisterPermission} from '/@/plugin/permission/index';
// @ts-ignore
import eIconPicker, { iconList, analyzingIconForIconfont } from 'e-icon-picker';
import 'e-icon-picker/icon/default-icon/symbol.js'; //基本彩色图标库
import 'e-icon-picker/index.css'; // 基本样式，包含基本图标
import 'font-awesome/css/font-awesome.min.css';
import elementPlus from 'e-icon-picker/icon/ele/element-plus.js'; //element-plus的图标
import fontAwesome470 from 'e-icon-picker/icon/fontawesome/font-awesome.v4.7.0.js'; //fontAwesome470的图标
import eIconList from 'e-icon-picker/icon/default-icon/eIconList.js';
import iconfont from '/@/assets/iconfont/iconfont.json'; //引入json文件
import '/@/assets/iconfont/iconfont.css'; //引入css
import '/@/assets/iconfont/iconfont-01/iconfont.css'; //引入css
import '/@/assets/iconfont/iconfont-02/iconfont.css'; //引入css
import VXETable from 'vxe-table'
import 'vxe-table/lib/style.css'

import '/@/assets/style/reset.scss';
import 'element-tree-line/dist/style.css'

let forIconfont = analyzingIconForIconfont(iconfont); //解析class
iconList.addIcon(forIconfont.list); // 添加iconfont dvadmin3的icon
iconList.addIcon(elementPlus); // 添加element plus的图标
iconList.addIcon(fontAwesome470); // 添加fontAwesome 470版本的图标

let app = createApp(App);

app.use(eIconPicker, {
	addIconList: eIconList, //全局添加图标
	removeIconList: [], //全局删除图标
	zIndex: 3100, //选择器弹层的最低层,全局配置
});

pinia.use(piniaPersist);
directive(app);
other.elSvg(app);


app.use(VXETable)
app.use(pinia)
	.use(router)
	.use(ElementPlus, { i18n: i18n.global.t })
	.use(i18n)
	.use(VueGridLayout)
	.use(fastCrud)
	.mount('#app');

app.config.globalProperties.mittBus = mitt();

/* ============================================================================
 * 平台定制（与 dvadmin 模板无关，整块可删）
 * ==========================================================================*/

// ① 标签栏/路由里的"首页"只保留一个。
//    历史上首页被两个来源各注入过一次（framework 的 utils/menu.ts 与前端菜单 store），
//    而且它是 isAffix=true 的固定标签（没有 ×），所以「两个首页」关不掉。这里启动时去重。
// 【根治】直接把持久化的路由/标签缓存整个清掉。
// 路由守卫的逻辑是「routesList 为空 → 立刻按当前后端菜单重新初始化」，
// 所以清空之后，任何历史遗留的旧首页路由（如早期的 platformHome）都会被彻底丢弃，
// 不再依赖对存储形状的猜测。只需清一次，之后就正常了。
['routesList', 'tagsViewRoutes', 'requestOldRoutes', 'keepAliveNames'].forEach((key) => {
	[Session, Local].forEach((store) => {
		if (store.get(key)) {
			store.remove(key);
			console.info(`[platform] 已清空缓存的路由数据：${key}`);
		}
	});
});

const isHomeLike = (item: any) =>
	item?.path === '/home' || item?.path === '/platform/home' ||
	item?.name === 'home' || item?.name === 'platformHome' || item?.meta?.title === '首页';
['tagsViewRoutes', 'routesList'].forEach((key) => {
	[Session, Local].forEach((store) => {
		const value = store.get(key);
		if (!value) return;

		// 去重函数：命中的"首页"条目只保留第一个
		const dedupeHome = (arr: any[]) => {
			let homeSeen = false;
			return arr.filter((item: any) => {
				if (!isHomeLike(item)) return true;
				if (homeSeen) return false;
				homeSeen = true;
				return true;
			});
		};
		// 递归找数组（pinia 持久化存的可能是整个 state 对象，而不是裸数组）
		const walk = (node: any, depth = 0): any => {
			if (depth > 4 || node === null || typeof node !== 'object') return node;
			if (Array.isArray(node)) {
				const cleaned = dedupeHome(node);
				return cleaned.length === node.length ? cleaned : cleaned;
			}
			const out: any = Array.isArray(node) ? [] : { ...node };
			let changed = false;
			Object.keys(node).forEach((k) => {
				const before = node[k];
				const after = walk(before, depth + 1);
				out[k] = after;
				if (after !== before) changed = true;
			});
			return changed ? out : node;
		};
		const cleanedValue = walk(value);
		if (cleanedValue !== value) {
			store.set(key, cleanedValue);
			console.info(`[platform] 首页标签去重：${key}`);
		}
	});
});

// ② 切换平台模块时整页刷新（省事做法）：每次进页面都拿最新数据，
//    不依赖 KeepAlive / onMounted 时序。只在"平台内部路由"之间跳转时触发，
//    并且带标志位，避免与"登录后 router.push('/home')"互相触发造成死循环。
const isPlatformRoute = (path: string) => path === '/home' || path.startsWith('/platform/');
let platformReloading = false;
router.afterEach((to, from) => {
	if (platformReloading || !to.path || !from.path) return;
	if (to.path === from.path) return;
	if (!isPlatformRoute(to.path) || !isPlatformRoute(from.path)) return;
	platformReloading = true;
	window.location.reload();
});

