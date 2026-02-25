# backend.py

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# -------------------
# 1. LLM
# -------------------
api_key = os.getenv("GOOGLE_API_KEY")

llm = init_chat_model(
    "google_genai:gemini-2.5-flash-lite",
    api_key=api_key
)

# -------------------
# 2. Tools
# -------------------

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL')."""

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing stock with Human-In-The-Loop approval.
    """

    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase declined by human.",
        }


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 5. Graph
# -------------------
memory = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat")

# Proper conditional routing
graph.add_conditional_edges(
    "chat",
    tools_condition,  # routes to "tools" if tool call exists
)

graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=memory)

# -------------------
# 6. CLI Example
# -------------------
if __name__ == "__main__":

    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Handle Interrupt
        while "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0]
            prompt_to_human = interrupt_data.value

            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ")

            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        # Print assistant response
        print("Bot:", result["messages"][-1].content)
        print()