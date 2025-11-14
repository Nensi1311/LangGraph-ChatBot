from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import requests
import atexit

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
DATABASE_URL = os.environ.get("POSTGRES_DB_URL")

if not DATABASE_URL:
    raise ValueError("POSTGRES_DB_URL environment variable is missing in .env file.")

# OpenRouter LLM
llm = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    # max_tokens=2000
)

# Built-in tool
search_tool = DuckDuckGoSearchRun(region="us-en")

# Custom calculator tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }
    except Exception as e:
        return {"error": str(e)}

# Combine tools
tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)


# Chat state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Chat node logic
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Tool node
tool_node = ToolNode(tools)

# Initialize persistent PostgresSaver (production-safe)
checkpointer_context = PostgresSaver.from_conn_string(DATABASE_URL)
checkpointer = checkpointer_context.__enter__()

try:
    checkpointer.setup()
except Exception as e:
    print(f"PostgresSaver setup warning: {e}")


# Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


# Compile graph with checkpointer
chatbot = graph.compile(checkpointer=checkpointer)

# Retrieve all chat threads
def retrieve_all_threads():
    all_threads = set()
    try:
        for checkpoints in checkpointer.list(config=None):
            all_threads.add(checkpoints.config["configurable"]["thread_id"])
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return list(all_threads)


# Cleanly close the Postgres connection on exit
@atexit.register
def close_checkpointer():
    try:
        checkpointer_context.__exit__(None, None, None)
        print("PostgresSaver connection closed cleanly.")
    except Exception as e:
        print(f"Error closing PostgresSaver: {e}")
