# app/frontend/customer_ui.py
"""
基于 Gradio 的 C 端用户界面 - v4.0
支持真实登录、多用户切换、横向越权测试
"""
import gradio as gr
import requests
import json
import time
import os
from typing import List, Dict, Optional, Tuple
from gradio import themes

# 配置
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


class ChatClient:
    """聊天客户端 - 支持真实登录"""
    
    def __init__(self, token: str, user_id: int, username: str):
        self.token = token
        self.user_id = user_id
        self.username = username
        self.thread_id = f"gradio_{username}_{int(time.time())}"
        print(f"✅ 客户端已初始化:  用户={username}, ID={user_id}")
    
    def send_message(self, message: str) -> Tuple[bool, str, dict]:
        """发送消息到 Agent"""
        if not message.strip():
            return False, "消息不能为空", {}
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        try:
            print(f"📤 [{self.username}] 发送消息: {message}")
            
            response = requests.post(
                f"{API_BASE_URL}/chat",
                headers=headers,
                json={
                    "question": message,
                    "thread_id": self.thread_id
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"API 错误 {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail.get('detail', response.text)}"
                except:
                    error_msg += f": {response.text[: 200]}"
                return False, error_msg, {}
            
            # 流式接收
            full_answer = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'token' in data:
                                full_answer += data['token']
                            elif 'error' in data:
                                return False, f"Agent 错误: {data['error']}", {}
                        except json.JSONDecodeError:
                            pass
            
            # 检查状态
            status_info = self.check_status()
            return True, full_answer, status_info
            
        except Exception as e:
            return False, f"请求失败: {str(e)}", {}
    
    def check_status(self) -> dict:
        """检查会话状态"""
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.get(
                f"{API_BASE_URL}/status/{self.thread_id}",
                headers=headers,
                timeout=10
            )
            return response.json() if response.status_code == 200 else {}
        except: 
            return {}


def login_user(username: str, password: str) -> Tuple[bool, str, Optional[ChatClient], str]:
    """
    用户登录
    
    Returns:
        (success, message, client, user_info_text)
    """
    if not username or not password:
        return False, "❌ 请输入用户名和密码", None, ""
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/login",
            json={"username":  username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            client = ChatClient(
                token=data["access_token"],
                user_id=data["user_id"],
                username=data["username"]
            )
            
            user_info = f"""
**登录成功！**

- 👤 用户名: {data['username']}
- 🆔 用户ID: {data['user_id']}
- 📛 姓名: {data['full_name']}
- 🛡️ 权限: {'管理员' if data['is_admin'] else '普通用户'}
            """
            
            return True, "✅ 登录成功", client, user_info
        else:
            error = response.json().get('detail', '登录失败')
            return False, f"❌ {error}", None, ""
            
    except Exception as e: 
        return False, f"❌ 登录失败:  {str(e)}", None, ""


def create_chat_interface():
    """创建聊天界面 - v4.0 优化版"""
    
    # 1. 定义更现代的主题
    theme = themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size=themes.sizes.radius_lg,
        font=[themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
    ).set(
        body_background_fill="#f8fafc",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    )

    # 2. 深度定制 CSS
    custom_css = """
    /* 全局布局调整 */
    footer {display: none !important;}
    .gradio-container {max-width: 1200px !important; margin: 0 auto;}
    
    /* 登录页样式 */
    .login-wrapper {
        max-width: 420px; 
        margin: 60px auto; 
        padding: 40px !important; 
        background: white; 
        border-radius: 16px; 
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
    }
    .login-header {text-align: center; margin-bottom: 24px;}
    .login-logo {font-size: 48px; margin-bottom: 10px;}
    
    /* 顶部导航栏 */
    .nav-header {
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background: white; 
        padding: 12px 24px; 
        border-radius: 12px; 
        border: 1px solid #e2e8f0; 
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .user-pill {
        background: #eff6ff; 
        color: #1e40af; 
        padding: 4px 12px; 
        border-radius: 999px; 
        font-size: 0.85em; 
        font-weight: 600;
        border: 1px solid #dbeafe;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* 状态卡片 */
    .audit-card { padding: 16px; border-radius: 8px; margin-top: 8px; font-size: 0.9em; border-left: 4px solid transparent; }
    .audit-card-pending { background: #fffbeb; border-color: #f59e0b; color: #92400e; }
    .audit-card-approved { background: #f0fdf4; border-color: #22c55e; color: #166534; }
    .audit-card-rejected { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
    
    /* 状态栏微调 */
    .status-badge {
        font-size: 0.75rem; 
        padding: 2px 8px; 
        border-radius: 4px; 
        display: inline-block;
        margin-top: 4px;
    }
    """
    
    with gr.Blocks(title="Smart Agent v4.0", theme=theme, css=custom_css) as demo:
        
        # 状态存储
        client_state = gr.State(None)
        
        # ====================
        #  登录界面 
        # ====================
        with gr.Group(visible=True) as login_panel:
            with gr.Column(elem_classes="login-wrapper"):
                gr.HTML("""
                <div class="login-header">
                    <div class="login-logo"></div>
                    <h2 style="margin:0; color:#1e293b;">欢迎登录</h2>
                    <p style="color:#64748b; margin-top:4px;">E-commerce Smart Agent v4.0</p>
                </div>
                """)
                
                with gr.Group():
                    username_input = gr.Textbox(
                        label="账号", 
                        placeholder="请输入用户名 (如 alice)", 
                        scale=1
                    )
                    password_input = gr.Textbox(
                        label="密码", 
                        type="password", 
                        placeholder="请输入密码", 
                        scale=1
                    )
                
                login_btn = gr.Button("立即登录", variant="primary", size="lg")
                login_message = gr.Markdown("", elem_id="login-msg")
                
                # 将测试账号信息折叠，保持界面整洁
                with gr.Accordion("假如你是开发者，点击查看测试账号", open=False):
                    gr.Markdown("""
                    | 用户 | 账号/密码 | 拥有订单 |
                    |---|---|---|
                    | **Alice** | `alice` / `alice123` | SN20240001-003 |
                    | **Bob** | `bob` / `bob123` | SN20240004-005 |
                    | **Admin** | `admin` / `admin123` | 管理员 |
                    """)

        # ====================
        #  聊天主界面
        # ====================
        with gr.Group(visible=False) as chat_panel:
            
            # 顶部导航栏
            with gr.Group(elem_classes="nav-header"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4, min_width=200):
                        gr.Markdown("###  智能客服助手", elem_classes="m-0")
                    
                    with gr.Column(scale=4, min_width=200):
                        # 这里用 HTML 动态显示用户信息
                        user_header_display = gr.HTML('<div class="user-pill"> 未登录</div>')
                    
                    with gr.Column(scale=1, min_width=100):
                        logout_btn = gr.Button("退出", size="sm", variant="secondary")

            with gr.Row():
                # 左侧：聊天区 (占宽 75%)
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="对话记录",
                        height=550,
                        placeholder="有什么可以帮您？尝试问问订单状态或退货政策。",
                        avatar_images=("https://ui-avatars.com/api/?name=User&background=random", "https://ui-avatars.com/api/?name=Bot&background=0D8ABC&color=fff"),
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="请输入消息... (Enter 发送)",
                            scale=5,
                            container=False,
                            autofocus=True
                        )
                        submit_btn = gr.Button( variant="primary", scale=1, min_width=60)
                    
                    status_display = gr.HTML("")

                # 右侧：功能面板 (占宽 25%)
                with gr.Column(scale=1):
                    gr.Markdown("###  快捷工具箱")
                    
                    with gr.Accordion(" 订单查询", open=True):
                        btn_query_own = gr.Button("我的订单", size="sm")
                        btn_query_alice = gr.Button("Alice 的订单", size="sm")
                        btn_query_bob = gr.Button("Bob 的订单", size="sm")
                        gr.Markdown("*用于测试越权访问*", elem_classes="text-xs text-gray-400")
                    
                    with gr.Accordion(" 售后服务", open=True):
                        btn_policy = gr.Button("退货政策", size="sm")
                        btn_refund = gr.Button("模拟: 尺码不合退货", size="sm")
                        btn_refund_high = gr.Button("模拟: 大额退款(触发风控)", size="sm")
                    
                    gr.Markdown("---")
                    clear_btn = gr.Button(" 清空历史", variant="stop", size="sm")

        # === 逻辑函数 ===
        
        def handle_login(username, password):
            success, message, client, user_info = login_user(username, password)
            if success:
                # 提取姓名用于 Header 显示
                name = client.username
                header_html = f'''
                <div style="display:flex; justify-content:flex-end; align-items:center;">
                    <span class="user-pill">👤 {name} (ID: {client.user_id})</span>
                </div>
                '''
                return (
                    client, 
                    gr.update(visible=False), # 隐藏登录
                    gr.update(visible=True),  # 显示聊天
                    header_html,
                    "", "", # 清空输入框
                    gr.Info("登录成功！") # 使用 Gradio 内置通知
                )
            else:
                return (None, gr.update(visible=True), gr.update(visible=False), "", username, password, gr.Warning(message))

        def handle_logout():
            return (
                None, 
                gr.update(visible=True), 
                gr.update(visible=False), 
                '<div class="user-pill">👤 未登录</div>',
                [], # 清空 Chatbot
                ""
            )

        def render_audit_card_v2(status_info: dict) -> str:
            """优化的审核卡片渲染"""
            status = status_info.get("status", "UNKNOWN")
            data = status_info.get("data", {})
            
            if status == "WAITING_ADMIN":
                return f'''
                <div class="audit-card audit-card-pending">
                    <b>⏳ 触发风控审核</b><br>
                    原因：{data.get("trigger_reason", "未知")}<br>
                    风险等级：{data.get("risk_level", "NORMAL")}
                </div>'''
            elif status == "APPROVED":
                return '<div class="audit-card audit-card-approved"> <b>审核通过</b><br>退款流程已启动</div>'
            elif status == "REJECTED":
                return f'<div class="audit-card audit-card-rejected"> <b>审核拒绝</b><br>{data.get("admin_comment", "无理由")}</div>'
            return ""

        def send_and_update_v2(message, history, client):
            """适配 Gradio 4.0 messages 格式的消息处理"""
            if not client:
                gr.Warning("会话已过期，请重新登录")
                yield history, message, ""
                return

            if not message.strip():
                yield history, message, ""
                return
            
            # 立即上屏用户消息
            history.append({"role": "user", "content": message})
            yield history, "", '<span class="status-badge" style="background:#e0f2fe; color:#0369a1;">Thinking...</span>'
            
            success, response, status_info = client.send_message(message)
            
            if not success:
                history.append({"role": "assistant", "content": f"❌ Error: {response}"})
                yield history, "", '<span class="status-badge" style="background:#fee2e2; color:#b91c1c;">Error</span>'
                return
            
            # 处理回复内容
            final_content = response
            status = status_info.get("status", "PROCESSING")
            
            # 追加漂亮的 HTML 卡片
            if status in ["WAITING_ADMIN", "APPROVED", "REJECTED"]:
                final_content += render_audit_card_v2(status_info)
            
            history.append({"role": "assistant", "content": final_content})
            
            status_text = "Ready"
            status_color = "#dcfce7; color:#15803d" # Green
            if status == "WAITING_ADMIN":
                status_text = "Waiting Audit"
                status_color = "#fef3c7; color:#b45309" # Yellow
            
            yield history, "", f'<span class="status-badge" style="background:{status_color};">📡 {status_text}</span>'

        # === 绑定事件 ===
        login_btn.click(
            handle_login,
            inputs=[username_input, password_input],
            outputs=[client_state, login_panel, chat_panel, user_header_display, username_input, password_input, login_message]
        )
        
        logout_btn.click(
            handle_logout,
            outputs=[client_state, login_panel, chat_panel, user_header_display, chatbot, login_message]
        )
        
        # 回车提交与按钮提交
        msg_input.submit(
            send_and_update_v2,
            inputs=[msg_input, chatbot, client_state],
            outputs=[chatbot, msg_input, status_display]
        )
        submit_btn.click(
            send_and_update_v2,
            inputs=[msg_input, chatbot, client_state],
            outputs=[chatbot, msg_input, status_display]
        )
        
        clear_btn.click(lambda: [], outputs=[chatbot])
        
        # 快捷按钮逻辑
        btn_query_own.click(lambda: "查询我的订单", outputs=msg_input)
        btn_query_alice.click(lambda: "查询订单 SN20240001", outputs=msg_input)
        btn_query_bob.click(lambda: "查询订单 SN20240004", outputs=msg_input)
        btn_policy.click(lambda: "内衣可以退货吗？", outputs=msg_input)
        btn_refund.click(lambda: "我要退货，订单号 SN20240003，尺码不合适", outputs=msg_input)
        btn_refund_high.click(lambda: "我要退款 2500 元，订单 SN20240003，质量有问题", outputs=msg_input)

    return demo


if __name__ == "__main__":
    print("🚀 启动 E-commerce Smart Agent v4.0 客户端界面...")
    print(f"📡 API 地址: {API_BASE_URL}")
    
    demo = create_chat_interface()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )