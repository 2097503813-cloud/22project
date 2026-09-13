/**
 * 平台业务接口专用的 axios 实例
 *
 * 为什么不用 /@/utils/service：
 *   那个实例是给 dvadmin 的 Django 接口用的，响应拦截器会校验 `{code, data, msg}` 信封，
 *   遇到裸 JSON 会抛 `非标准返回：[object Object]` 并把请求 reject —— 我们的
 *   model_service（Flask）业务接口返回的就是裸 JSON（/models、/trainings、/predict…），
 *   所以 5 个页面全部报错。这里单独建一个实例：直接返回 response.data，不做信封校验。
 *
 * 与框架保持一致的地方：baseURL 同样取 VITE_API_URL；token 同样放 Authorization 头。
 */
import axios from 'axios';
import { Session } from '/@/utils/storage';

const platformRequest = axios.create({
	baseURL: import.meta.env.VITE_API_URL as string,
	timeout: 180000, // 训练/推理是同步阻塞的，给足时间
	headers: { 'Content-Type': 'application/json' },
});

platformRequest.interceptors.request.use((config) => {
	const token = Session.get('token');
	if (token) config.headers!['Authorization'] = `${token}`;
	return config;
});

platformRequest.interceptors.response.use(
	(response) => response.data, // 裸 JSON 直接返回，不校验信封
	(error) => {
		const data = error?.response?.data;
		const msg = (data && (data.error || data.msg)) || error?.message || '请求失败';
		const err = new Error(msg);
		(err as any).payload = data;
		(err as any).status = error?.response?.status;
		return Promise.reject(err);
	}
);

export default platformRequest;
