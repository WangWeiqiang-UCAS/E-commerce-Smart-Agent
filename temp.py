# app/frontend/customer_ui.py
"""
基于 Gradio 的 C 端用户界面 - v4.0
支持实时卡片渲染和状态同步
"""
import gradio as gr
import requests
import json
import time
import os
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from gradio import themes

# 配置
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
DEFAULT_USER_ID = 1  # 默认用户ID，实际应用需要登录


class ChatClient:
    """聊天客户端"""
    
    def __init__(self, user_id: int = DEFAULT_USER_ID):
        self.user_id = user_id
        self.thread_id = f"gradio_user_{user_id}_{int(time.time())}"
        self.token = None
        self._init_token()
    
    def _init_token(self):
        """初始化 Token（简化版，实际需要登录接口）"""
        # 临时方案：直接生成 Token
        from app.core.security import create_access_token
        self.token = create_access_token(user_id=self.user_id, is_admin=False)
        print(f" Token 已生成:  {self.token[: 20]}...")
    
    def send_message(self, message: str) -> Tuple[bool, str, dict]:
        """
        发送消息到 Agent
        
        Returns:
            (success, response_text, status_info)
        """
        if not message.strip():
            return False, "消息不能为空", {}
        
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        try:
            print(f" 发送消息: {message}")
            print(f" API:  {API_BASE_URL}/chat")
            
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
            
            print(f" 响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"API 错误 {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail.get('detail', response.text)}"
                except:
                    error_msg += f": {response.text[: 200]}"
                print(f" {error_msg}")
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
                                print(f" Agent 错误: {data['error']}")
                                return False, f"Agent 错误: {data['error']}", {}
                        except json.JSONDecodeError:
                            pass
            
            print(f" 收到回复: {full_answer[: 100]}...")
            
            # 检查状态
            status_info = self.check_status()
            
            return True, full_answer, status_info
            
        except requests.exceptions.Timeout:
            error_msg = "请求超时，请稍后重试"
            print(f" {error_msg}")
            return False, error_msg, {}
        except requests.exceptions.ConnectionError:
            error_msg = f"无法连接到服务器 ({API_BASE_URL})，请检查服务是否启动"
            print(f" {error_msg}")
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"请求失败: {str(e)}"
            print(f" {error_msg}")
            import traceback
            traceback.print_exc()
            return False, error_msg, {}
    
    def check_status(self) -> dict:
        """检查当前会话状态"""
        headers = {
            "Authorization": f"Bearer {self.token}",
        }
        
        try: 
            response = requests.get(
                f"{API_BASE_URL}/status/{self.thread_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                status_data = response.json()
                print(f" 状态: {status_data.get('status')}")
                return status_data
            else:
                print(f" 状态查询失败: {response.status_code}")
                return {"status": "ERROR", "message": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f" 状态查询异常: {e}")
            return {"status": "ERROR", "message": str(e)}


def create_chat_interface():
    """创建聊天界面"""
    
    # 自定义 CSS
    custom_css = """
    . audit-card-pending {
        background:  linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius:  12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
    }
    .audit-card-approved {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    .audit-card-rejected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    .status-bar {
        padding: 12px;
        border-radius: 8px;
        margin:  8px 0;
        font-weight: 500;
    }
    . status-ready {
        background-color: #d4edda;
        color:  #155724;
        border:  1px solid #c3e6cb;
    }
    .status-processing {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .status-waiting {
        background-color: #fff3cd;
        color:  #856404;
        border: 1px solid #ffeaa7;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    """
    
    with gr.Blocks(
        title="E-commerce Smart Agent v4.0", 
        theme=themes.Soft(),
        css=custom_css
    ) as demo:
        
        gr.Markdown("#  E-commerce Smart Agent v4.0")
        gr.Markdown("### 全栈·沉浸式人机协作系统 | The Immersive System")
        
        # 状态存储
        client_state = gr.State(None)
        
        with gr.Row():
            # 左侧：聊天区
            with gr.Column(scale=3):
                # 使用新版 Chatbot 格式
                chatbot = gr.Chatbot(
                    label=" 对话窗口",
                    height=500,
                    avatar_images=(
                        "https://api.dicebear.com/7.x/avataaars/svg?seed=user",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=agent"
                    )
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="",
                        placeholder="请输入您的问题（支持订单查询、政策咨询、退货申请）.. .",
                        scale=4,
                        lines=2,
                        max_lines=5
                    )
                    submit_btn = gr.Button(" 发送", variant="primary", scale=1, size="lg")
                
                # 状态显示区
                status_display = gr.HTML(
                    value='<div class="status-bar status-ready"> 状态: 就绪</div>'
                )
            
            # 右侧：功能区
            with gr.Column(scale=1):
                gr.Markdown("### ⚡ 快捷操作")
                
                with gr.Group():
                    btn_query_order = gr.Button(" 查询订单", size="sm", variant="secondary")
                    btn_policy = gr.Button(" 退货政策", size="sm", variant="secondary")
                    btn_refund_normal = gr.Button(" 申请退货", size="sm", variant="secondary")
                    btn_refund_high = gr.Button(" 大额退款测试", size="sm", variant="secondary")
                
                gr.Markdown("---")
                gr.Markdown("### 🔧 系统信息")
                
                with gr.Group():
                    user_id_display = gr.Textbox(
                        label="用户ID",
                        value=str(DEFAULT_USER_ID),
                        interactive=False
                    )
                    thread_id_display = gr.Textbox(
                        label="会话ID",
                        interactive=False,
                        value=""
                    )
                    api_status = gr.Textbox(
                        label="API 状态",
                        value="未检测",
                        interactive=False
                    )
                
                gr.Markdown("---")
                clear_btn = gr.Button(" 清空对话", variant="stop", size="sm")
        
        # === 功能函数 ===
        
        def init_client():
            """初始化客户端"""
            try:
                client = ChatClient(user_id=DEFAULT_USER_ID)
                
                # 测试 API 连接
                try:
                    health_response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
                    if health_response.status_code == 200:
                        api_status_text = " 已连接"
                    else:
                        api_status_text = f" 异常 ({health_response.status_code})"
                except:
                    api_status_text = " 无法连接"
                
                return client, client.thread_id, api_status_text
            except Exception as e:
                print(f" 初始化失败: {e}")
                return None, "初始化失败", f" 错误: {str(e)}"
        
        def render_audit_card(status_info: dict) -> str:
            """渲染审核卡片"""
            status = status_info.get("status", "UNKNOWN")
            data = status_info.get("data", {})
            message = status_info.get("message", "")
            
            if status == "WAITING_ADMIN":
                risk_level = data.get("risk_level", "UNKNOWN")
                trigger_reason = data.get("trigger_reason", "无")
                
                return f'''
                <div class="audit-card-pending">
                    <h3 style="margin:  0 0 12px 0; color: #856404;"> 正在人工审核</h3>
                    <p style="margin: 6px 0;"><strong>风险等级: </strong> <span style="color: #d39e00;">{risk_level}</span></p>
                    <p style="margin: 6px 0;"><strong>触发原因:</strong> {trigger_reason}</p>
                    <p style="margin: 12px 0 0 0; font-size: 0.9em; color: #856404;">
                         我们将在 24 小时内完成审核，请耐心等待。您可以关闭页面，稍后返回查看结果。
                    </p>
                </div>
                '''
            
            elif status == "APPROVED": 
                admin_comment = data.get("admin_comment", "")
                
                return f'''
                <div class="audit-card-approved">
                    <h3 style="margin: 0 0 12px 0; color: #155724;"> 审核已通过</h3>
                    <p style="margin: 6px 0;">您的申请已通过审核，正在处理中...</p>
                    {f'<p style="margin: 6px 0;"><strong>审核意见:</strong> {admin_comment}</p>' if admin_comment else ''}
                    <p style="margin: 12px 0 0 0; font-size:  0.9em; color: #155724;">
                         资金将在 3-5 个工作日内原路退回，请注意查收。
                    </p>
                </div>
                '''
            
            elif status == "REJECTED":
                admin_comment = data.get("admin_comment", "请联系客服")
                
                return f'''
                <div class="audit-card-rejected">
                    <h3 style="margin:  0 0 12px 0; color: #721c24;"> 审核未通过</h3>
                    <p style="margin: 6px 0;"><strong>原因:</strong> {admin_comment}</p>
                    <p style="margin: 12px 0 0 0; font-size: 0.9em; color: #721c24;">
                         如有疑问，请联系在线客服或拨打客服热线。
                    </p>
                </div>
                '''
            
            return ""
        
        def send_and_update(message: str, history: List[Dict], client: ChatClient):
            """发送消息并更新界面 - 使用新版 messages 格式"""
            if not client: 
                return (
                    history,
                    message,
                    '<div class="status-bar status-error"> 客户端未初始化</div>'
                )
            
            if not message.strip():
                return (
                    history,
                    message,
                    '<div class="status-bar status-error"> 请输入消息</div>'
                )
            
            # 添加用户消息 - 新版格式
            history.append({
                "role": "user",
                "content": message
            })
            
            # 更新状态：处理中
            status_html = '<div class="status-bar status-processing"> 正在思考... </div>'
            
            # 先返回一次，显示用户消息和处理状态
            yield history, "", status_html
            
            # 发送请求
            success, response, status_info = client.send_message(message)
            
            if not success:
                # 发送失败 - 添加助手错误消息
                history.append({
                    "role": "assistant",
                    "content": f" 发送失败: {response}"
                })
                status_html = '<div class="status-bar status-error">❌ 发送失败</div>'
                yield history, "", status_html
                return
            
            # 检查是否需要审核
            status = status_info.get("status", "PROCESSING")
            
            # 构建助手回复内容
            assistant_content = response
            
            if status == "WAITING_ADMIN":
                # 插入审核卡片
                card_html = render_audit_card(status_info)
                assistant_content += "\n\n" + card_html
                status_html = '<div class="status-bar status-waiting"> 等待人工审核中... </div>'
                
            elif status == "APPROVED": 
                # 审核通过卡片
                card_html = render_audit_card(status_info)
                assistant_content += "\n\n" + card_html
                status_html = '<div class="status-bar status-ready"> 审核已通过</div>'
                
            elif status == "REJECTED": 
                # 审核拒绝卡片
                card_html = render_audit_card(status_info)
                assistant_content += "\n\n" + card_html
                status_html = '<div class="status-bar status-ready">审核未通过</div>'
                
            else:
                # 正常回复
                status_html = '<div class="status-bar status-ready"> 就绪</div>'
            
            # 添加助手回复 - 新版格式
            history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            yield history, "", status_html
        
        def clear_chat():
            """清空对话"""
            return [], '<div class="status-bar status-ready"> 对话已清空</div>'
        
        def set_example_message(example:  str):
            """设置示例消息"""
            return example
        
        # === 事件绑定 ===
        
        # 页面加载时初始化
        demo.load(
            init_client,
            outputs=[client_state, thread_id_display, api_status]
        )
        
        # 发送消息
        submit_btn.click(
            send_and_update,
            inputs=[msg_input, chatbot, client_state],
            outputs=[chatbot, msg_input, status_display]
        )
        
        msg_input.submit(
            send_and_update,
            inputs=[msg_input, chatbot, client_state],
            outputs=[chatbot, msg_input, status_display]
        )
        
        # 清空对话
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, status_display]
        )
        
        # 快捷按钮
        btn_query_order.click(
            set_example_message,
            inputs=gr.State("查询订单 SN20240001"),
            outputs=msg_input
        )
        
        btn_policy.click(
            set_example_message,
            inputs=gr.State("内衣可以退货吗？"),
            outputs=msg_input
        )
        
        btn_refund_normal.click(
            set_example_message,
            inputs=gr.State("我要退货，订单号 SN20240003，尺码不合适"),
            outputs=msg_input
        )
        
        btn_refund_high.click(
            set_example_message,
            inputs=gr.State("我要退款 2500 元，订单 SN20240003，质量有问题"),
            outputs=msg_input
        )
    
    return demo


if __name__ == "__main__":
    print(" 启动 E-commerce Smart Agent v4.0 客户端界面...")
    print(f" API 地址: {API_BASE_URL}")
    print(f" 用户ID: {DEFAULT_USER_ID}")
    
    demo = create_chat_interface()
    demo.queue()  # 启用队列以支持流式输出
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )