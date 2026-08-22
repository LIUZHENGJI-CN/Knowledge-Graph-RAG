# academic_frontend.py

import streamlit as st

from router import answer_question


# ============================================================
# 1. Streamlit 页面配置
# ============================================================

st.set_page_config(
    page_title="学术知识图谱问答系统",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# 2. 辅助函数
# ============================================================

def get_route_name(route):
    """
    将 router 返回的路线名称转换成适合页面展示的中文。
    """

    route_names = {
        "fast_rag": "语义文献问答",
        "text2cypher": "知识图谱查询",
        "hybrid": "混合查询",
    }

    return route_names.get(
        route,
        route or "未知"
    )


def show_sources(sources):
    """
    展示 Fast RAG 返回的参考论文。
    """

    if not sources:
        return

    with st.expander("📄 参考论文"):

        for i, source in enumerate(sources, 1):

            title = source.get("title") or "Unknown Title"

            entry_id = (
                source.get("entry_id")
                or source.get("id")
            )

            score = source.get("score")
            methods = source.get("methods")
            keywords = source.get("keywords")

            st.markdown(
                f"**{i}. {title}**"
            )

            if entry_id:
                st.caption(
                    f"Paper ID：{entry_id}"
                )

            if score is not None:
                try:
                    st.caption(
                        f"向量相似度：{float(score):.4f}"
                    )
                except Exception:
                    pass

            if methods and methods != "未提及":
                st.write(
                    f"**方法：** {methods}"
                )

            if keywords and keywords != "无":
                st.write(
                    f"**关键词：** {keywords}"
                )

            if i < len(sources):
                st.divider()


def show_graph_details(result):
    """
    展示 Text2Cypher / Hybrid 返回的图谱查询信息。
    """

    cypher = result.get("cypher")
    data = result.get("data")

    if not cypher and not data:
        return

    with st.expander("🔎 图谱查询详情"):

        if cypher:

            st.markdown(
                "**生成的 Cypher 查询**"
            )

            st.code(
                cypher,
                language="cypher"
            )

        if data:

            st.markdown(
                "**Neo4j 查询结果**"
            )

            try:

                st.dataframe(
                    data,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception:

                st.json(
                    data
                )


def run_question(question):
    """
    调用整个后端系统。

    流程：

    用户问题
        ↓
    router.answer_question()
        ↓
    Router 自动选择：
        fast_rag
        text2cypher
        hybrid
        ↓
    返回最终回答
    """

    try:

        result = answer_question(
            question
        )

        return result

    except Exception as e:

        return {
            "answer": f"系统运行失败：{e}",
            "route": None,
        }


# ============================================================
# 3. 页面标题
# ============================================================

st.title(
    "📚 学术知识图谱问答系统"
)

st.caption(
    "基于 Neo4j、向量检索与大语言模型的学术智能问答"
)

st.divider()


# ============================================================
# 4. 初始化聊天记录
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 5. 首次进入页面时显示提示
# ============================================================

if not st.session_state.messages:

    st.info(
        "你可以直接输入学术问题，"
        "系统会自动选择知识图谱查询或语义文献问答。"
    )

    st.markdown(
        """
**例如：**

- 什么是分子网络定量建模？
- 寻找与 Ilya Nemenman 合作最频繁的作者
- 找深度学习方向的专家
- 复旦大学的作者主要研究什么关键词？
"""
    )


# ============================================================
# 6. 显示历史聊天记录
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # 如果是 AI 消息，
        # 还可以显示 Router 返回的额外信息
        if message["role"] == "assistant":

            result = message.get(
                "result"
            )

            if result:

                route = result.get(
                    "route"
                )

                if route:

                    st.caption(
                        f"查询方式：{get_route_name(route)}"
                    )

                # Fast RAG 的论文来源
                show_sources(
                    result.get("sources")
                )

                # Text2Cypher / Hybrid 的图谱信息
                show_graph_details(
                    result
                )


# ============================================================
# 7. 用户输入框
# ============================================================

question = st.chat_input(
    "请输入你的问题..."
)


# ============================================================
# 8. 用户提交问题
# ============================================================

if question:

    # --------------------------------------------------------
    # 保存用户消息
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # --------------------------------------------------------
    # 显示用户消息
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # 调用 Router
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "正在检索知识图谱并生成回答..."
        ):

            result = run_question(
                question
            )


        # ----------------------------------------------------
        # 获取 Router 返回结果
        # ----------------------------------------------------

        answer = result.get(
            "answer",
            "系统没有生成回答。"
        )

        route = result.get(
            "route"
        )


        # ----------------------------------------------------
        # 显示最终回答
        # ----------------------------------------------------

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # 显示 Router 选择路线
        # ----------------------------------------------------

        if route:

            st.caption(
                f"查询方式：{get_route_name(route)}"
            )


        # ----------------------------------------------------
        # 显示 Fast RAG 参考论文
        # ----------------------------------------------------

        show_sources(
            result.get("sources")
        )


        # ----------------------------------------------------
        # 显示图谱查询详情
        # ----------------------------------------------------

        show_graph_details(
            result
        )


    # --------------------------------------------------------
    # 保存 AI 回答
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "result": result,
        }
    )

# streamlit run frontend.py
