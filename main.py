from agent.graph import build_graph


def main():
    app = build_graph()

    # 测试
    result = app.invoke({
        "user_input": "我头痛3天了，吃布洛芬能缓解吗",
        "retry_count": 0,
    })

    print("=" * 50)
    print(f"科室：{result.get('department')}")
    print(f"紧急度：{result.get('urgency')}")
    print(f"是否建议就医：{result.get('should_see_doctor')}")
    print(f"安全审查：{'通过' if result.get('is_safe') else '未通过'}")
    print(f"最终回复：{result.get('final_response')}")


if __name__ == "__main__":
    main()