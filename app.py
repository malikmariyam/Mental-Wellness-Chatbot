import os
import uuid
import streamlit as st
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Set your NVIDIA API key
os.environ["NVIDIA_API_KEY"] = ""

# Initialize the model
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

# Define system prompt
system_prompt = """You are a warm, empathetic, and non-judgmental virtual mental wellness assistant.
You support users with emotional well-being using CBT, mindfulness, and journaling.
Avoid giving diagnoses or medical advice.

If the user mentions self-harm:
"I'm here to support you, but it’s important to talk to someone trained to help. Please contact a mental health professional or call a crisis line like 988 if you're in the U.S."

Examples:
- Emotional reflection: “I’m feeling anxious lately…”
- Journaling: “Let’s write a gratitude journal.”
- Reframing: “I had a bad day and feel worthless.”"""

# Set theme and layout
st.set_page_config(page_title="🧠 AI-Powered Psychology Chatbot", layout="wide")
st.title("🧠 AI-Powered Psychology Chatbot")

# Chat session persistence
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}  # id: memory
if "chat_names" not in st.session_state:
    st.session_state.chat_names = {}  # id: name (Chat A, Chat B, ...)
if "current_session" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_session = new_id
    st.session_state.chat_sessions[new_id] = ConversationBufferMemory()
    st.session_state.chat_names[new_id] = "Chat A"

# Sidebar - session selector
with st.sidebar:
    st.markdown("### 💬 Chat Settings")
    if st.button("🆕 Start New Chat"):
        new_id = str(uuid.uuid4())
        next_letter = chr(65 + len(st.session_state.chat_names))  # 'A', 'B', 'C', ...
        st.session_state.current_session = new_id
        st.session_state.chat_sessions[new_id] = ConversationBufferMemory()
        st.session_state.chat_names[new_id] = f"Chat {next_letter}"

    # Chat session dropdown
    session_keys = list(st.session_state.chat_sessions.keys())
    if session_keys:
        selected = st.selectbox(
            "Previous Chats",
            options=session_keys,
            format_func=lambda x: st.session_state.chat_names.get(x, f"Chat {x[:6]}")
        )
        st.session_state.current_session = selected

# Active memory
active_memory = st.session_state.chat_sessions[st.session_state.current_session]

# Chain using active memory
conversation = ConversationChain(
    llm=llm,
    memory=active_memory,
    verbose=True,
    output_parser=StrOutputParser(),
    prompt=PromptTemplate(
        input_variables=["history", "input"],
        template=system_prompt + "\n\n{history}\nUser: {input}"
    )
)

# Chat input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Prevent consecutive duplicate messages
    if not active_memory.chat_memory.messages or user_input != active_memory.chat_memory.messages[-1].content:
        result = conversation.invoke({"input": user_input})
        response_text = result["response"] if isinstance(result, dict) else str(result)

        # Add to memory
        active_memory.chat_memory.add_user_message(user_input)
        active_memory.chat_memory.add_ai_message(response_text)

# Display conversation
last_displayed = {"user": None, "ai": None}
for msg in active_memory.chat_memory.messages:
    if msg.type == "human":
        if msg.content != last_displayed["user"]:
            st.chat_message("user").write(msg.content)
            last_displayed["user"] = msg.content
    elif msg.type == "ai":
        if msg.content != last_displayed["ai"]:
            st.chat_message("assistant").write(msg.content)
            last_displayed["ai"] = msg.content

