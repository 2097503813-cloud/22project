/**
 * @description 校验是否为租户模式。租户模式把域名替换成 域名 加端口
 */
export const getBaseURL = function (url: null | string = null, isHost: null | boolean = null) {
	let baseURL = import.meta.env.VITE_API_URL as any;
	// 如果需要host返回，时，返回地址前缀加http地址
	if (isHost && !baseURL.startsWith('http')) {
		baseURL = window.location.protocol + '//' + window.location.host + baseURL
	}
	if (url) {
		const regex = /^(http|https):\/\//;
		if (regex.test(url)) {
			return url
		} else {
			// js判断是否是斜杠结尾
			return baseURL.replace(/\/$/, '') + '/' + url.replace(/^\//, '');
		}
	}
	if (!baseURL.endsWith('/')) {
		baseURL += '/';
	}
	return baseURL;
};

export const getWsBaseURL = function () {
	let baseURL = import.meta.env.VITE_API_URL as any;
	let param = baseURL.split('/')[3] || '';
	if (param !== '' || baseURL.startsWith('/')) {
		baseURL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.hostname + (location.port ? ':' : '') + location.port + baseURL;
	}
	if (!baseURL.endsWith('/')) {
		baseURL += '/';
	}
	if (baseURL.startsWith('http')) {
		// https 也默认会被替换成 wss
		baseURL = baseURL.replace('http', 'ws');
	}
	return baseURL;
};
