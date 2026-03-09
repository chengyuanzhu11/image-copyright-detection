/**
 * 图像版权检测系统 - 通用登录状态管理模块
 * 统一处理所有页面的登录状态检查和用户身份验证
 */

// 检查登录状态的通用函数
function checkLogin() {
    console.log('检查登录状态...');
    const user = localStorage.getItem('currentUser');
    
    // 获取DOM元素
    const authOverlay = document.getElementById('authOverlay');
    const authContainer = document.getElementById('authContainer');
    const loginTab = document.getElementById('loginTab');
    const registerTab = document.getElementById('registerTab');
    const authFrame = document.getElementById('authFrame');
    const appContent = document.getElementById('appContent');
    const welcomeAlert = document.getElementById('welcomeAlert');
    const welcomeUser = document.getElementById('welcomeUser');
    const currentUser = document.getElementById('currentUser');
    
    if (!authOverlay || !appContent) {
        console.error('找不到认证或应用内容元素！');
        return;
    }
    
    if (user) {
        // 用户已登录，显示应用内容
        console.log('用户已登录:', user);
        
        // 提取JSON对象中的用户名（如果是JSON对象）
        let displayName = user;
        try {
            const userObj = JSON.parse(user);
            if (userObj && userObj.username) {
                displayName = userObj.username;
            }
        } catch (e) {
            // 如果不是JSON，就使用原始值（即用户名字符串）
            console.log('用户数据不是JSON格式，直接显示用户名');
        }
        
        // 更新用户显示
        if (currentUser) currentUser.textContent = displayName;
        if (welcomeUser) welcomeUser.textContent = displayName;
        
        // 平滑过渡动画
        authOverlay.style.opacity = '0';
        setTimeout(() => {
            authOverlay.style.display = 'none';
            appContent.style.display = 'block';
            
            // 显示欢迎提示
            if (welcomeAlert) {
                welcomeAlert.style.display = 'block';
                
                // 5秒后隐藏欢迎提示
                setTimeout(() => {
                    welcomeAlert.style.display = 'none';
                }, 5000);
            }
        }, 500);
    } else {
        // 用户未登录，显示认证界面
        console.log('用户未登录，显示登录界面');
        
        if (authContainer) authContainer.style.display = 'block';
        authOverlay.style.display = 'flex';
        authOverlay.style.opacity = '1';
        appContent.style.display = 'none';
        if (welcomeAlert) welcomeAlert.style.display = 'none';
        
        // 仅在iframe尚未加载登录/注册页面时设置src，
        // 避免在用户填写表单时重新加载iframe导致输入丢失
        if (authFrame) {
            var currentSrc = authFrame.src || '';
            var isOnAuthPage = currentSrc.indexOf('login.html') !== -1 || 
                               currentSrc.indexOf('register.html') !== -1;
            if (!isOnAuthPage) {
                authFrame.src = 'login.html';
                if (loginTab) loginTab.classList.add('active');
                if (registerTab) registerTab.classList.remove('active');
            }
        }
    }
}

// 登出处理函数
function handleLogout() {
    console.log('执行登出操作...');
    
    // 清除登录状态
    localStorage.removeItem('currentUser');
    localStorage.removeItem('userToken');
    localStorage.removeItem('lastActivity');
    
    // 重新检查登录状态
    checkLogin();
    console.log('登出完成，已重新检查登录状态');
}

// 处理登录成功消息
function handleLoginMessage(data) {
    console.log('处理登录成功消息:', data);
    
    // 保存用户信息到localStorage
    if (data.username) {
        localStorage.setItem('currentUser', data.username);
        console.log('已保存用户信息:', data.username);
    }
    
    if (data.token) {
        localStorage.setItem('userToken', data.token);
        console.log('已保存用户令牌:', data.token);
    }
    
    localStorage.setItem('lastActivity', Date.now());
    console.log('已更新最后活动时间');
    
    // 重新检查登录状态，刷新UI
    console.log('准备刷新登录状态...');
    setTimeout(() => {
        checkLogin();
        console.log('已刷新登录状态UI');
    }, 100);
}

// 设置页面加载时的登录状态检查
document.addEventListener('DOMContentLoaded', function() {
    console.log('页面加载完成，检查登录状态');
    
    // 检查登录状态
    checkLogin();
    
    // 为登出按钮添加事件监听
    const logoutBtn = document.getElementById('navLogoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
    
    // 为登录和注册标签添加点击事件 - 这部分功能已移到index.html中直接通过onclick事件处理
    // 保留这段代码是为了向后兼容，但主要交互已通过内联事件处理程序实现
    const loginTab = document.getElementById('loginTab');
    const registerTab = document.getElementById('registerTab');
    const authFrame = document.getElementById('authFrame');
    
    if (loginTab && registerTab && authFrame) {
        console.log('发现auth标签，准备添加备用事件监听器');
        
        // 这里注释掉直接事件添加，改用代理方式，避免与内联事件冲突
        /* 
        loginTab.addEventListener('click', function() {
            loginTab.classList.add('active');
            registerTab.classList.remove('active');
            
            // 使用淡出效果切换
            authFrame.style.opacity = '0';
            setTimeout(() => {
                authFrame.src = 'login.html';
                authFrame.onload = function() {
                    authFrame.style.opacity = '1';
                };
            }, 300);
            
            console.log('切换到登录页面');
        });
        
        registerTab.addEventListener('click', function() {
            registerTab.classList.add('active');
            loginTab.classList.remove('active');
            
            // 使用淡出效果切换
            authFrame.style.opacity = '0';
            setTimeout(() => {
                authFrame.src = 'register.html';
                authFrame.onload = function() {
                    authFrame.style.opacity = '1';
                };
            }, 300);
            
            console.log('切换到注册页面');
        });
        */
        
        // 使用事件代理在父元素上监听点击事件
        const authTabs = document.querySelector('.auth-tabs');
        if (authTabs) {
            console.log('添加标签容器的事件代理');
            authTabs.addEventListener('click', function(e) {
                // 确保点击的是标签，而不是其他元素
                let tabElement = e.target;
                while (tabElement && !tabElement.classList.contains('auth-tab')) {
                    if (tabElement === authTabs) {
                        // 没有找到标签元素，退出
                        return;
                    }
                    tabElement = tabElement.parentElement;
                }
                
                if (!tabElement) return;
                
                console.log('通过事件代理捕获到标签点击:', tabElement.id);
                if (tabElement.id === 'loginTab') {
                    // 执行切换到登录的全局函数
                    if (typeof switchToLogin === 'function') {
                        switchToLogin();
                    } else {
                        console.log('switchToLogin函数不存在，使用备用方法');
                        loginTab.classList.add('active');
                        registerTab.classList.remove('active');
                        authFrame.src = 'login.html';
                    }
                } else if (tabElement.id === 'registerTab') {
                    // 执行切换到注册的全局函数
                    if (typeof switchToRegister === 'function') {
                        switchToRegister();
                    } else {
                        console.log('switchToRegister函数不存在，使用备用方法');
                        registerTab.classList.add('active');
                        loginTab.classList.remove('active');
                        authFrame.src = 'register.html';
                    }
                }
            });
        }
    }
    
    // 添加消息监听
    window.addEventListener('message', function(event) {
        // 验证消息源
        if (event.origin === window.location.origin) {
            const data = event.data;
            
            if (data && data.type === 'login-success') {
                handleLoginMessage(data);
            } else if (data && data.type === 'show-register') {
                // 显示注册标签
                console.log('收到切换到注册页面的消息');
                const registerTab = document.getElementById('registerTab');
                if (registerTab) {
                    // 首先尝试调用全局函数
                    if (typeof switchToRegister === 'function') {
                        switchToRegister();
                    } else {
                        // 备用方法
                        registerTab.click();
                    }
                }
            } else if (data && data.type === 'show-login') {
                // 显示登录标签
                console.log('收到切换到登录页面的消息');
                const loginTab = document.getElementById('loginTab');
                if (loginTab) {
                    // 首先尝试调用全局函数
                    if (typeof switchToLogin === 'function') {
                        switchToLogin();
                    } else {
                        // 备用方法
                        loginTab.click();
                    }
                }
            } else if (data && data.type === 'register-success') {
                // 注册成功后自动切换到登录标签
                console.log('收到注册成功消息，将切换到登录页面');
                setTimeout(function() {
                    const loginTab = document.getElementById('loginTab');
                    if (loginTab) {
                        // 首先尝试调用全局函数
                        if (typeof switchToLogin === 'function') {
                            switchToLogin();
                        } else {
                            // 备用方法
                            loginTab.click();
                        }
                    }
                }, 1000);
            }
        }
    });
    
    // 检查会话超时
    setInterval(function() {
        const lastActivity = localStorage.getItem('lastActivity');
        if (lastActivity) {
            const now = Date.now();
            const timeSinceLastActivity = now - parseInt(lastActivity);
            
            // 如果超过30分钟没有活动，自动登出
            if (timeSinceLastActivity > 30 * 60 * 1000) {
                console.log('会话超时，自动登出');
                handleLogout();
            }
        }
    }, 60 * 1000); // 每分钟检查一次
}); 