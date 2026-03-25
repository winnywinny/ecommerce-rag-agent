import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go

# === 🌟 核心升级：引入 LangChain 工业级生态 ===
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 页面与全局样式配置
# ==========================================
st.set_page_config(page_title="全域电商智能大屏 | LangChain版", page_icon="📈", layout="wide")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f6f8fd 0%, #f1f5f9 100%);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 全域电商数据可视化与 LangChain 决策大脑")
st.markdown("基于 **LangChain + FAISS** 构建的工业级 RAG 架构，支持海量数据大屏与企业级内部知识库注入。")

# ==========================================
# 2. 全局配置与状态初始化
# ==========================================
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://ws-vvr85dv3ndbxfxtu.ap-southeast-1.maas.aliyuncs.com/compatible-model/v1"

# 使用 Session State 缓存向量数据库，防止页面刷新导致重复计算
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ==========================================
# 3. 侧边栏：多平台切换与【LangChain 知识库挂载】
# ==========================================
st.sidebar.header("🎯 经营大盘控制台")

# --- LangChain RAG 知识库上传区 ---
st.sidebar.markdown("### 📚 内部知识库 (LangChain FAISS)")
knowledge_file = st.sidebar.file_uploader("上传公司运营SOP (TXT格式)", type=["txt"])

if knowledge_file is not None and st.session_state.vectorstore is None:
    with st.spinner("🧠 正在使用 LangChain 构建 FAISS 密集向量知识库..."):
        try:
            raw_text = knowledge_file.read().decode('utf-8')
            
            # 1. 智能文档切块 (避免切断整句话)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200, 
                chunk_overlap=30, # 上下文重叠，保留语义连贯性
                separators=["\n\n", "\n", "。", "！", "？", "，", ""]
            )
            chunks = text_splitter.split_text(raw_text)
            
            # 2. 调用 Embedding 模型将其转化为高维向量
            embeddings = OpenAIEmbeddings(
                openai_api_key=API_KEY, 
                openai_api_base=BASE_URL,
                model="text-embedding-v1" # 阿里云兼容的通用向量模型
            )
            
            # 3. 存入 FAISS 本地向量数据库
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.sidebar.success(f"✅ FAISS 知识库已挂载！(共 {len(chunks)} 个高维向量)")
        except Exception as e:
            st.sidebar.error(f"构建知识库失败，请检查 Embedding 配置。错误：{e}")

if st.session_state.vectorstore is not None:
    st.sidebar.success("✅ FAISS 向量数据库：运行中")

st.sidebar.markdown("---")
# --- 数据源选择区 ---
st.sidebar.markdown("### 📊 数据源配置")
logo_map = {
    "淘宝": "https://img.alicdn.com/tfs/TB1_uT8a5ERMeJjSspiXXbZLFXa-143-59.png",
    "京东": "https://misc.360buyimg.com/lib/img/e/logo-201305-b.png",
    "拼多多": "拼多多.png", 
    "1688": "阿里巴巴.png",
    "苏宁易购": "苏宁易购.png"
}
logo_placeholder = st.sidebar.empty() 
platforms = ["淘宝", "京东", "拼多多", "1688", "苏宁易购", "🛠️ 自定义数据上传"]
selected_platform = st.sidebar.radio("请选择分析数据源", platforms)

# 数据加载引擎
@st.cache_data
def load_and_enhance_data(platform_name):
    file_path = f"{platform_name}.csv"
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except Exception:
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
            df = pd.DataFrame({
                '商品类别': np.random.choice(['美妆', '服饰', '数码', '食品', '家居', '运动'], 1000),
                '消费金额': np.random.uniform(50, 3000, 1000),
                '购买时间': np.random.choice(dates, 1000),
                '用户城市': np.random.choice(['北京', '上海', '广州', '深圳', '成都', '杭州'], 1000),
                '用户性别': np.random.choice(['男', '女'], 1000, p=[0.4, 0.6]),
                '用户年龄': np.random.randint(18, 60, 1000)
            })
    df['购买时间'] = pd.to_datetime(df['购买时间'])
    df['日期'] = df['购买时间'].dt.date
    df['小时'] = df['购买时间'].dt.hour
    df['星期'] = df['购买时间'].dt.day_name()
    return df

if selected_platform != "🛠️ 自定义数据上传":
    if selected_platform in logo_map:
        logo_placeholder.image(logo_map[selected_platform], width=120)
    with st.spinner(f"正在抽取 {selected_platform} 业务数据..."):
        df = load_and_enhance_data(selected_platform)
else:
    logo_placeholder.empty()
    uploaded_file = st.sidebar.file_uploader("📂 请上传电商交易流水 (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except Exception:
            df = pd.read_csv(uploaded_file, encoding='gbk')
        if '购买时间' in df.columns:
            df['购买时间'] = pd.to_datetime(df['购买时间'])
            df['日期'] = df['购买时间'].dt.date
            df['小时'] = df['购买时间'].dt.hour
            df['星期'] = df['购买时间'].dt.day_name()
        else:
            st.error("❌ 上传的 CSV 必须包含【购买时间】列！")
            st.stop()
    else:
        st.info("👈 请上传 CSV 文件以开启分析。")
        st.stop()

real_orders = len(df)
real_gmv = df['消费金额'].sum() if '消费金额' in df.columns else 0
avg_price = real_gmv/real_orders if real_orders>0 else 0
mock_uv = real_orders * np.random.randint(15, 25) 
mock_pv = mock_uv * 3.5
mock_cart = int(mock_uv * 0.3)

# ==========================================
# 4. UI 布局：四大业务模块 Tab 切换
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 交易与流量", "👥 用户与商品", "🚀 营销与供应链", "🧠 LangChain 深度诊断"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 销售额 GMV", f"¥ {real_gmv:,.0f}")
    c2.metric("💳 支付订单量", f"{real_orders:,} 笔")
    c3.metric("🛒 客单价", f"¥ {avg_price:,.2f}")
    c4.metric("👥 独立访客(UV)", f"{mock_uv:,} 人")
    
    col_t1, col_t2 = st.columns([6, 4])
    with col_t1:
        if '日期' in df.columns and '消费金额' in df.columns:
            daily_sales = df.groupby('日期')['消费金额'].sum().reset_index()
            fig_trend = px.area(daily_sales, x='日期', y='消费金额', title="📈 每日 GMV 趋势", color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig_trend, use_container_width=True)
    with col_t2:
        fig_funnel = go.Figure(go.Funnel(
            y=['浏览', '访客', '加购', '下单', '支付'], x=[mock_pv, mock_uv, mock_cart, int(real_orders*1.2), real_orders],
            textinfo="value+percent initial", marker={"color": ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]}
        ))
        fig_funnel.update_layout(title="🔽 核心转化漏斗分析")
        st.plotly_chart(fig_funnel, use_container_width=True)

with tab2:
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        if '商品类别' in df.columns and '消费金额' in df.columns:
            cat_sales = df.groupby('商品类别')['消费金额'].sum().reset_index()
            fig_tree = px.treemap(cat_sales, path=['商品类别'], values='消费金额', title="🛍️ 商品品类营收贡献")
            st.plotly_chart(fig_tree, use_container_width=True)
    with col_u2:
        if '用户年龄' in df.columns and '用户性别' in df.columns:
            fig_demo = px.histogram(df, x='用户年龄', color='用户性别', barmode='group', title="👥 年龄与性别分布")
            st.plotly_chart(fig_demo, use_container_width=True)

with tab3:
    st.info("💡 宏观供应链与营销效率监控大屏。")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_pie = px.pie(names=['搜索', '推荐', '直播', '广告', '自然'], values=[35, 25, 20, 15, 5], hole=0.5, title="🌐 流量来源结构")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_m2:
        fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = 98.2, title = {'text': "📦 48小时发货率 (%)"}))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ----------------- TAB 4: LangChain RAG 核心流水线 -----------------
with tab4:
    st.markdown(f"### 🧠 基于 LangChain 的企业级商业诊断")
    if st.session_state.vectorstore:
        st.success("🎯 知识检索增强 (RAG) 已激活：AI 将严格遵循 FAISS 向量库中的内部规则进行诊断。")
    else:
        st.warning("⚠️ RAG 离线模式：未检测到内部知识库，AI 将使用世界通用知识输出报告。")
    
    if st.button("✨ 启动 LangChain 诊断链路 ✨", type="primary"):
        with st.spinner("LangChain Agent 正在流转数据与知识库..."):
            
            # 1. 提取大盘摘要特征
            cat_top = df.groupby('商品类别')['消费金额'].sum().sort_values(ascending=False).head(2).index.tolist() if '商品类别' in df.columns else ["未知"]
            city_top = df.groupby('用户城市')['消费金额'].sum().sort_values(ascending=False).head(3).index.tolist() if '用户城市' in df.columns else ["未知"]
            
            summary_text = f"""
            当前平台：{selected_platform}
            核心指标：总销售额 ¥{real_gmv:,.0f}，客单价 ¥{avg_price:,.0f}。
            用户画像：核心城市集中在 {", ".join(city_top)}。
            商品画像：高转化品类是 {", ".join(cat_top)}。
            """
            
            # 2. RAG 向量检索：带着业务特征去知识库里“搜索”
            context_text = "未挂载知识库，请根据通用电商经验进行诊断。"
            if st.session_state.vectorstore:
                # 构造符合当前业务场景的检索词
                search_query = f"针对客单价 {avg_price} 的用户，以及高优城市 {city_top[0]} 和品类 {cat_top[0]}，公司有什么运营规范和话术？"
                
                # 初始化检索器 (寻找匹配度最高的 3 个切片)
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(search_query)
                
                if docs:
                    context_text = "\n".join([f"- {doc.page_content}" for doc in docs])
                    st.info(f"🔍 **FAISS 语义检索命中条目**：\n{context_text}")
            
            # 3. 初始化 LLM 和 LangChain Prompt
            llm = ChatOpenAI(
                model="qwen-plus",
                openai_api_key=API_KEY,
                openai_api_base=BASE_URL,
                temperature=0.3 # 调低温度，让 AI 更加严谨、不胡说八道
            )
            
            prompt_template = """你是一位年薪百万的电商大厂数据运营总监。
            请根据以下【实时大盘数据】，结合【公司内部运营知识库】（如果有），出具分析建议。

            【实时大盘数据】：
            {data_summary}

            【内部运营知识库检索结果】：
            {context}

            请严格按以下结构输出（使用 Markdown，多用业务黑话和emoji）：
            1. 📈 **大盘一句话总结**
            2. 🎯 **人货场运营策略**（必须严格遵守【内部运营知识库】中的规章制度，若未提供知识库则凭借经验分析）
            3. 💡 **具体执行 Action**
            """
            
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["data_summary", "context"]
            )
            
            # 4. 构建 LCEL (LangChain Expression Language) 流水线
            chain = prompt | llm | StrOutputParser()
            
            try:
                # 5. 执行流水线并输出
                response = chain.invoke({
                    "data_summary": summary_text, 
                    "context": context_text
                })
                st.markdown("---")
                st.markdown(response)
            except Exception as e:
                st.error(f"LangChain 调用链路失败，请检查网络或 API 配置。错误详情：{e}")
