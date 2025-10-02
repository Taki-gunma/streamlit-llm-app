import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env読み込み
load_dotenv()

# OpenAI APIキー
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# モデル準備
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

# 専門家の種類（選択肢）
experts = {
    "歴史学者": "あなたは歴史の専門家です。歴史的背景や出来事をわかりやすく説明してください。",
    "医師": "あなたは医療の専門家です。健康や病気について正確かつ丁寧に答えてください。",
    "エンジニア": "あなたはエンジニアです。技術的な課題を解決するアドバイスをしてください。",
}

# LLM応答を生成する関数
def generate_answer(user_input: str, expert: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", experts[expert]),
        ("human", "{question}")
    ])
    chain = template | llm | StrOutputParser()
    return chain.invoke({"question": user_input})


# --- Streamlit UI ---
st.title("🤖 LangChain × Streamlit (Lesson8風)")

st.write("""
### 使い方
1. 入力フォームに質問や相談を入力してください  
2. 専門家を選んで「送信」を押すと、その専門家として回答します  
""")

# ラジオボタン
expert_choice = st.radio("専門家を選んでください:", list(experts.keys()))

# 入力フォーム
with st.form("form"):
    user_text = st.text_area("質問内容を入力してください")
    submitted = st.form_submit_button("送信")

# 回答表示
if submitted and user_text.strip():
    st.write("### 回答")
    st.success(generate_answer(user_text, expert_choice))
