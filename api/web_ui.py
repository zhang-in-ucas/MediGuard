"""Gradio Web UI：多Agent + RAG + 安全审查"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from agent.graph import build_graph


def chat(message, history):
    """处理用户输入，返回Agent回复"""
    if not message.strip():
        return history, ""

    app = build_graph()

    # 拼接对话历史
    chat_history = []
    for msg in history:
        chat_history.append(msg["content"])

    result = app.invoke({
        "user_input": message,
        "retry_count": 0,
        "chat_history": chat_history,
    })

    department = result.get("department", "全科")
    urgency = result.get("urgency", "medium")
    should_see = result.get("should_see_doctor", False)
    is_safe = result.get("is_safe", True)
    final = result.get("final_response", "")

    urgency_label = {"high": "🔴 高", "medium": "🟡 中", "low": "🟢 低"}.get(urgency, "🟡 中")

    # 拼接用户可见内容
    process = ""
    if not is_safe:
        process += f"🚫 **安全审查未通过**，该问题涉及用药指导，建议咨询专业医生。\n\n"
        process += final
    else:
        process += f"📋 建议前往科室：**{department}** | 紧急度：{urgency_label}"
        if should_see:
            process += " | ⚠️ 建议就医"
        process += "\n\n---\n\n"
        process += final

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": process})
    return history, ""


def create_ui():
    with gr.Blocks(title="MediGuard 合规医疗安全问诊助手") as demo:
        gr.Markdown(
            "# 🛡️ MediGuard 合规医疗安全问诊助手\n"
            "基于多Agent协作 + RAG知识库 + 双层记忆 + 法规合规审查的医疗健康咨询系统\n\n"
            "⚠️ 本系统仅提供健康参考，不开展诊疗活动、不替代执业医师。"
        )
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            msg = gr.Textbox(
                placeholder="请描述您的症状，如：我头痛3天了，吃布洛芬能缓解吗",
                scale=4,
                show_label=False,
            )
            submit = gr.Button("发送", scale=1, variant="primary")
        gr.Examples(
            examples=[
                "我头痛3天了，吃布洛芬能缓解吗",
                "感冒了多喝水有用吗",
                "我发烧38.5度，该怎么办",
                "最近总是胸闷气短是怎么回事",
                "小孩咳嗽一周了还没好怎么办",
                "我血压150/95，说明什么？",
                "皮肤过敏可以吃氯雷他定吗",
                "运动后腿酸怎么缓解",
            ],
            inputs=msg,
        )
        submit.click(chat, [msg, chatbot], [chatbot, msg])
        msg.submit(chat, [msg, chatbot], [chatbot, msg])
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True
    )