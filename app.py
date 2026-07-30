import streamlit as st
import requests
import json

# 页面配置
st.set_page_config(page_title="☕ Mocha AI 咖啡推荐", layout="wide")

st.title("☕ Mocha AI 智能咖啡推荐系统")
st.caption("基于 RAG + 混合检索的语义推荐引擎")

# 侧边栏
with st.sidebar:
    st.header("🔍 搜索选项")
    query = st.text_input("输入你的咖啡需求", placeholder="例如：冷萃、酸度低、巧克力味...")
    top_k = st.slider("推荐数量", min_value=1, max_value=5, value=3)
    search_btn = st.button("🚀 搜索", type="primary")

# 主区域
if search_btn and query:
    with st.spinner("正在为你寻找最合适的咖啡..."):
        try:
            resp = requests.post(
                "http://localhost:8000/recommend_with_reason",
                json={"query": query, "top_k": top_k},
                timeout=10
            )
            data = resp.json()
            
            if data.get("results"):
                cols = st.columns(len(data["results"]))
                for idx, (col, result) in enumerate(zip(cols, data["results"])):
                    with col:
                        # 第 1 名特殊标注
                        if idx == 0:
                            st.markdown(f"### 🥇 推荐 #{idx+1}")
                        else:
                            st.markdown(f"### ☕ 推荐 #{idx+1}")
                        st.markdown(f"**{result['name']}**")
                        st.markdown(f"💰 ¥{result['price']}")
                        st.markdown(f"⚖️ 酸度: {result['acidity']}")
                        st.markdown(f"☕ 适合: {', '.join(result['suitable_for'])}")
                        st.markdown(f"📊 匹配度: {result['similarity']:.2f}")
                        # 仅第 1 名显示推荐语
                        if idx == 0 and result.get("reason"):
                            st.divider()
                            st.markdown(f"💬 **推荐理由**：{result['reason']}")
            else:
                st.warning("没有找到匹配的咖啡，试试其他关键词吧")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ 无法连接到后端服务，请确保 FastAPI 已启动 (uvicorn api.main:app)")
        except Exception as e:
            st.error(f"出错了: {str(e)}")
else:
    st.info("👆 在左侧输入你的需求，例如 '冷萃' 或 '低酸度 手冲' 试试")

# 底部
st.divider()
st.caption("💡 支持语义搜索：如'巧克力味'、'高酸度冰滴'、'性价比高的美式'")