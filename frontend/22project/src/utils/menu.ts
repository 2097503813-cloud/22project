import XEUtils from "xe-utils"
import {dynamicRoutes, staticRoutes} from "/@/router/route";

/**
 * @description: 处理后端菜单数据格式
 * @param {Array} menuData
 * @return {*}
 */
export const handleMenu = (menuData: Array<any>) => {
    // 先处理menu meta数据转换
    const handleMeta = (item: any) => {
        item.meta = {
            title: item.title,
            isLink: item.link_url,
            isHide: !item.visible,
            isKeepAlive: item.cache,
            isAffix: item.is_affix,
            isIframe: item.is_iframe,
            roles: ['admin'],
            icon: item.icon
        }
        item.name = item.component_name
        item.path = item.web_path
        return item
    }

    // 处理框架外的路由
    const handleFrame = (item: any) => {
        if (item.is_iframe) {
            item.meta = {
                title: item.title,
                isLink: item.link_url,
                isHide: !item.visible,
                isKeepAlive: item.cache,
                isAffix: item.is_affix,
                isIframe: item.is_iframe,
                roles: ['admin'],
                icon: item.icon
            }
            item.name = item.component_name
            item.path = item.web_path
        }
        return item
    }

    // 框架内路由
    const defaultRoutes:Array<any> = []
    // 框架外路由
    const iframeRoutes:Array<any> = []

    menuData.forEach((val) => {
        // if (val.is_iframe) {
        //     // iframeRoutes.push(handleFrame(val))
        // } else {
        //     defaultRoutes.push(handleMeta(val))
        // }
        defaultRoutes.push(handleMeta(val))
    })
    const data = XEUtils.toArrayTree(defaultRoutes, {
        parentKey: 'parent',
        strict: true,
    })
    const dynamicRoutes = [
        {
            // 注意：这里原本写死指向模板自带的 /system/home/index（标题走 i18n 的 message.router.home）。
            // 它和 stores/frontendMenu.ts 里那个 /home 是**两套路由模式各自注入的默认首页**，
            // 后端控制路由（isRequestRoutes=true）时生效的是这一份。
            // 两边都改成本平台自己的首页，才不会出现"两个首页"。
            path: '/home', name: 'home', component: '/platform/home/index', meta: {
                title: '首页',
                isLink: '',
                isHide: false,
                isKeepAlive: false,   // 不缓存：每次切回首页都重新挂载、重新拉 KPI
                isAffix: true,
                isIframe: false,
                roles: ['admin'],
                icon: 'ele-HomeFilled'
            }
        },
        ...data
    ]
    return {frameIn:dynamicRoutes,frameOut:iframeRoutes}
}
