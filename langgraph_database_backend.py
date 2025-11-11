from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()

# for openai api key
# llm = ChatOpenAI()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

# openrouter api key
llm = ChatOpenAI(
    api_key=openrouter_api_key, 
    base_url="https://openrouter.ai/api/v1", 
    model="google/gemini-2.5-flash",
    model_kwargs={"max_tokens": 3000} 
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect("langgraph_chatbot.db", check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoints in checkpointer.list(None):
        all_threads.add(checkpoints.config['configurable']['thread_id'])
    return list(all_threads)