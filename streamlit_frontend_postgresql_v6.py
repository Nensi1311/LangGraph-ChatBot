import streamlit as st
from langgraph_postgresql_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# Utility functions
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.toast("🆕 New chat created")

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        st.session_state["thread_names"][thread_id] = f"Chat {len(st.session_state['chat_threads'])}"

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# Session setup
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_names" not in st.session_state:
    st.session_state["thread_names"] = {
        tid: f"Chat {i+1}" for i, tid in enumerate(st.session_state["chat_threads"])
    }

if "renaming_thread" not in st.session_state:
    st.session_state["renaming_thread"] = None

add_thread(st.session_state["thread_id"])


# Sidebar UI
st.sidebar.title("🤖 LangGraph ChatBot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.header("💬 Chats")

for thread_id in st.session_state["chat_threads"][::-1]:
    chat_name = st.session_state["thread_names"].get(thread_id, f"Chat {thread_id}")

    col1, col2 = st.sidebar.columns([3, 1])

    # Load selected chat
    with col1:
        if st.button(chat_name, key=f"load_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})
            st.session_state["message_history"] = temp_messages
            st.toast(f"📂 Loaded chat: {chat_name}")

    # GPT-style 3-dot menu
    with col2:
        with st.popover("⋯", use_container_width=False):
            st.markdown("**Options**")

            # Rename option
            if st.button("✏️ Rename", key=f"rename_{thread_id}"):
                st.session_state["renaming_thread"] = thread_id
                st.rerun()

# Rename input UI
if st.session_state.get("renaming_thread"):
    tid = st.session_state["renaming_thread"]
    new_name = st.sidebar.text_input(
        "✏️ Enter new chat name:",
        value=st.session_state["thread_names"].get(tid, ""),
        key=f"rename_input_{tid}"
    )

    col_a, col_b = st.sidebar.columns([1, 1])
    with col_a:
        if st.button("✅ Save"):
            st.session_state["thread_names"][tid] = new_name or "Untitled Chat"
            st.session_state["renaming_thread"] = None
            st.toast("✅ Chat renamed successfully!")
            st.rerun()

    with col_b:
        if st.button("❌ Cancel"):
            st.session_state["renaming_thread"] = None
            st.rerun()


# Main UI
st.title("How Can I Assist You Today?")

if st.session_state["thread_id"]:
    thread_name = st.session_state["thread_names"].get(st.session_state["thread_id"], "Unnamed Chat")

    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.text(message["content"])

    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "Chat_Turn",
        }

        with st.chat_message("assistant"):
            status_holder = {"box": None}

            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="Tool finished", state="complete", expanded=False
                )

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
else:
    st.info("Select or create a chat to begin.")


# Styling (for GPT-style minimal sidebar)
st.markdown("""
<style>
button[kind="popover"] {
    font-size: 16px !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
