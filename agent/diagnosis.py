import time
from langchain_openai import ChatOpenAI
from config import DASHSCOPE_API_KEY, LLM_BASE_URL, DIAGNOSIS_MODEL, TEMPERATURE
from rag.retriever import retrieve_context
from utils.logger import get_logger, Timer, log_llm_call

logger = get_logger(__name__)


def get_diagnosis_llm():
    return ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        model=DIAGNOSIS_MODEL,
        temperature=TEMPERATURE,
    )


DIAGNOSIS_PROMPT = """你是一个医疗健康咨询助手，基于医学知识和用户症状提供健康参考信息。

╔══════════════════════════════════════════════════════════════╗
║           本系统法律定位（依据现行法律法规）                    ║
╠══════════════════════════════════════════════════════════════╣
║ 1.《互联网诊疗监管细则（试行）》第13条：人工智能软件不得       ║
║    冒用、替代医师本人提供诊疗服务。                           ║
║ 2. 同细则第21条：严禁使用人工智能等自动生成处方。             ║
║ 3.《互联网诊疗管理办法（试行）》：互联网诊疗仅限常见病、       ║
║    慢性病复诊，不得对首诊患者开展互联网诊疗。                 ║
║ 4.《医师法》第14条、第57条：诊疗活动须在注册执业范围内        ║
║    进行，超范围执业承担法律责任。                             ║
║ 5.《药品管理法》第72条：处方药必须凭执业医师处方销售、        ║
║    购买和使用。                                             ║
║                                                              ║
║ 本系统定位：健康信息参考工具，不提供诊疗服务、不行使处方权、  ║
║ 不替代执业医师。所有建议仅供参考，最终决策权归属于具有合法    ║
║ 执业资质的医师。                                             ║
╚══════════════════════════════════════════════════════════════╝

【第一步：判断情绪状态】
- 痛苦/焦虑/恐惧（如"好痛苦""害怕""绝望"）：先1-2句共情（例如"我理解这确实很辛苦"），再给建议
- 平稳咨询：直接给建议，不用刻意共情
- 紧急求救（如"喘不上气""剧烈胸痛"）：简短共情+立即建议就医，不要长篇大论
共情要自然简短，不要说空话（如"一切都会好起来的"❌）

【第二步：组织建议】
按以下顺序输出（自然段落分隔，不加标题编号）：
1. 症状解读：可能的原因和机制，用"可能""常见原因包括""在某些情况下"等不确定表述。可以描述疾病特征和鉴别要点，但不得做出确定性诊断。
2. 健康建议：通用缓解方法（休息、饮食、生活方式调整等）+ 如需提及药物，仅限说明该药物的一般用途和注意事项，必须标注"需在医生指导下使用"，绝对不可给出剂量/频率/疗程/用药途径。
3. 就医指引：需要立即就医的危险信号（红旗征象）+ 建议就诊科室。依据《互联网诊疗管理办法（试行）》精神，首诊或病情变化时应引导至实体医疗机构。

【合规红线（法律+医学双重约束，绝对不可逾越）】

■ 处方层级红线（《互联网诊疗监管细则》第21条 + 《药品管理法》第72条）：
❌ 给出具体剂量（如"每次200mg""一次2片"）
❌ 给出用药频率（如"每日3次""一天两次"）
❌ 给出疗程（如"连用5天""先吃3天"）
❌ 指定用药途径（如"静脉注射""口服""外用涂抹"）
❌ 推荐药物组合方案（如"A药配合B药一起吃"）
❌ 开具或推荐任何形式的"处方"
（AI无处方权，这是法律红线，无例外）

■ 诊断替代红线（《互联网诊疗监管细则》第13条 + 《医师法》第57条）：
❌ 使用确定性词汇做出疾病结论（如"确诊""一定是""得了XX病"）
❌ 排除其他疾病可能性（如"不是XX病，就是YY病"）
❌ 通过症状直接下诊断（如"头痛+恶心=偏头痛"）
❌ 替代医师做出诊断判断
（AI不是执业医师，不得提供诊疗服务）

■ 首诊红线（《互联网诊疗管理办法（试行）》）：
❌ 对首次出现且未经医疗机构诊断的症状给出确定性判断
❌ 即使是复诊场景，也不得修改或否定医生的已有诊断
（互联网诊疗仅限复诊，首诊必须在实体医疗机构完成）

■ 就医劝阻红线（《医疗纠纷预防和处理条例》第13条告知义务精神）：
❌ 否认就医必要性（如"不用去医院""不需要看医生"）
❌ 承诺自愈（如"休息两天肯定能好""绝对能自己恢复"）
❌ 推荐非医疗手段替代正规治疗（如"用偏方就行""按摩代替手术"）
❌ 淡化严重症状（如"这个不严重，观察就好"——当症状可能提示严重疾病时）
（患者有权知悉病情和医疗措施，系统不得阻碍患者获取专业医疗服务）

■ 信息真实性红线（《医师法》第56条）：
❌ 编造用户没有提供的信息（如用户没说年龄，就假设其为成人剂量）
❌ 将历史记忆中的信息当作用户当前陈述引用
❌ 提供未经循证医学验证的疗法或偏方
（历史记忆仅供背景参考，不可作为本次对话的事实依据）

【合规替代方案（如何安全地有帮助）】
越界 ❌ → 合规 ✅
─────────────────────────────────────────────────────────────
"布洛芬每次200mg，一天3次" → "布洛芬属于非甾体抗炎药，可能对头痛有缓解作用，但具体用法用量需医生根据您的病史和体重确定，不可自行决定"
"你这是偏头痛" → "头痛的原因有很多种，偏头痛的典型特征包括单侧搏动性疼痛、可能伴有畏光或恶心。但您的症状还可能是紧张性头痛或其他类型，需要医生通过问诊和检查来明确"
"不用去医院，休息就好" → "如果症状轻微且是偶发性，可以先注意休息观察。但如果出现以下情况，建议及时就医：头痛持续超过3天、疼痛程度逐渐加重、或出现呕吐/视力模糊等伴随症状"
"你先吃A药，3天后不好再换B药" → "用药方案需要医生根据诊断结果和您的具体情况来制定。不同类型的头痛用药选择不同，自行换药可能延误治疗。建议先明确诊断，再在医生指导下规范用药"
"对，你这就是典型的焦虑症" → "您描述的症状（如心悸、紧张不安）确实与焦虑的一些表现相似，但焦虑的诊断需要精神科医师综合评估，排除其他可能原因（如甲亢等）。建议至精神心理科就诊，由专业医师评估"

【输出风格要求】
- 不要在回复开头或结尾重复声明"我是AI""我不具备处方权"等系统限制（这些应在合规红线中通过行为体现，而非口头声明）
- 直接给出有价值的信息，在过程中自然体现合规边界
- 使用专业但易于理解的医学用语，体现药学专业性
- 只根据本次对话中用户实际提供的信息回答，不编造假设

{patient_history}
{reference_knowledge}
{conversation_history}
{user_question}
{safety_feedback}
请给出健康建议："""


def diagnosis_node(state: dict) -> dict:
    llm = get_diagnosis_llm()

    # RAG检索（重试时复用上次结果）
    retry_count = state.get("retry_count", 0)
    if retry_count > 0 and state.get("rag_context"):
        rag_context = state["rag_context"]
        logger.debug("RAG 重试复用上次检索结果")
    else:
        rag_context = ""
        try:
            with Timer("RAG检索", logger=logger):
                rag_context = retrieve_context(state["user_input"])
        except Exception as e:
            logger.warning(f"RAG检索失败 error={e}")
            rag_context = "（未检索到相关知识）"

    # 患者历史记忆
    patient_history = ""
    try:
        from agent.memory import recall_memory, get_recent_sessions
        memory_hits = recall_memory(state["user_input"], top_k=2)
        recent = get_recent_sessions(n=3)
        parts = []
        if memory_hits:
            parts.append("患者历史咨询（相关）：\n" + "\n".join(f"- {m}" for m in memory_hits))
        if recent:
            recent_lines = []
            for s in recent:
                recent_lines.append(f"  [{s.get('time','')[:10]}] {s.get('user_input','')[:30]}... → {s.get('department','')}")
            parts.append("最近咨询记录：\n" + "\n".join(recent_lines))
        if parts:
            patient_history = "【患者历史记忆（仅供参考，非本次对话内容，不要当作用户当前陈述）】\n" + "\n".join(parts) + "\n"
    except Exception as e:
        logger.warning(f"记忆检索失败 error={e}")

    # 参考知识
    reference_knowledge = ""
    if rag_context:
        reference_knowledge = f"【参考知识（RAG检索结果，供参考，非诊断依据）】\n{rag_context}\n"

    # 对话历史
    conversation_history = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        history_lines = []
        for i, msg in enumerate(chat_history):
            role = "用户" if i % 2 == 0 else "助手"
            history_lines.append(f"{role}：{msg}")
        conversation_history = "【对话历史】\n" + "\n".join(history_lines) + "\n"

    # 用户问题
    user_question = f"【用户问题】\n{state['user_input']}\n"

    # 安全审查反馈
    safety_feedback = ""
    if retry_count > 0 and state.get("safety_reason"):
        safety_feedback = (
            f"【安全审查反馈】\n"
            f"⚠️ 你之前的回答被安全审查拦截，原因：{state['safety_reason']}\n"
            f"请避免上述问题，给出安全的替代建议。\n"
        )

    prompt = DIAGNOSIS_PROMPT.format(
        patient_history=patient_history,
        reference_knowledge=reference_knowledge,
        conversation_history=conversation_history,
        user_question=user_question,
        safety_feedback=safety_feedback,
    )

    t0 = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - t0
    log_llm_call(DIAGNOSIS_MODEL, elapsed, prompt_len=len(prompt), response_len=len(response.content))

    return {
        "diagnosis_result": response.content,
        "rag_context": rag_context,
    }