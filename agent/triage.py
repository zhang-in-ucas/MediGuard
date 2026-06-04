from langchain_openai import ChatOpenAI
from config import DASHSCOPE_API_KEY, LLM_BASE_URL, TRIAGE_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

def get_triage_llm():
    return ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        model=TRIAGE_MODEL,
        temperature=0.3,
    )

TRIAGE_PROMPT = """你是一个医疗分诊员。根据用户输入判断科室和紧急程度。

注意：如果用户输入是对之前问题的补充（如补充年龄、症状细节等），应结合上下文判断，不要脱离之前的症状单独分诊。

{history_section}

用户输入：{user_input}

请输出JSON：
{{"department": "科室", "urgency": "high/medium/low", "should_see_doctor": true/false}}"""

def triage_node(state: dict) -> dict:
    llm = get_triage_llm()

    # 拼接对话历史
    history_section = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        history_lines = []
        for i, msg in enumerate(chat_history):
            role = "用户" if i % 2 == 0 else "助手"
            history_lines.append(f"{role}：{msg}")
        history_section = "对话历史：\n" + "\n".join(history_lines) + "\n\n"

    prompt = TRIAGE_PROMPT.format(
        history_section=history_section,
        user_input=state["user_input"],
    )

    import json
    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content)
        return {
            "department": result.get("department", "全科"),
            "urgency": result.get("urgency", "medium"),
            "should_see_doctor": result.get("should_see_doctor", True),
        }
    except Exception as e:
        logger.warning(f"分诊失败，使用默认值 error={e}")
        return {
            "department": "全科",
            "urgency": "medium",
            "should_see_doctor": True,
        }