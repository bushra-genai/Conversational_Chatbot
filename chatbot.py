import os
from dotenv import load_dotenv
import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

## Setup
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Groq Q&A conversational Chatbot with Memory", page_icon="🤖")

st.title("🤖 Groq Conversational Chatbot")

# Sidebar
with st.sidebar:
    st.subheader("Controls")
    model_name = st.selectbox(
        "Groq Model",
        ["deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.1-8b- instant"],
        index=2
    )

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150, 10)

system_prompt = st.text_area(
    "System Prompt (Rules)",
    value="You are a helpful concise teaching assistant. Use short, clear explanations"
)
st.caption("Tip: Lower Temperature for factual tasks; raise for brainstorming")

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar options for chat history
st.sidebar.header("💾 Chat History")

# ✅ Always show download button
history_json = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
st.sidebar.download_button(
    label="⬇️ Download Chat History",
    data=history_json if history_json else "[]",
    file_name="chat_history.json",
    mime="application/json"
)

# Clear button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.pop("history", None)
    st.session_state.pop("memory", None)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 Developed by **Bushra**")


if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Add it to your .env or deployment secret")

# Chat input
user_input = st.chat_input("Type your message .....")

# llm Setup
llm = ChatGroq(
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)

# Conversation Chain (classic)
conv = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False
)

if user_input:
    # Get response from model
    response = conv.predict(input=user_input)

    # Save to history
    st.session_state.history.append({"user": user_input, "bot": response})

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Display bot response
    with st.chat_message("assistant"):
        st.write(response)
