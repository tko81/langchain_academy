from langchain_community.chat_models.tongyi import ChatTongyi
from IPython.display import Image, display

from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import HumanMessage, SystemMessage


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

tools = [multiply]
llm = ChatTongyi(model="qwen3-coder-flash")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")


# no-op node that should be interrupted on
def human_feedback(state: MessagesState):
    # 这个节点会被 interrupt_before 拦截
    # 用户输入通过 API 的 update_state 方法提供
    # 这里只是一个占位符，实际的用户输入会通过外部 API 调用注入
    pass


# Assistant node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("human_feedback", human_feedback)

# Define edges: these determine the control flow
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")  # 工具执行后回到assistant

# 在 LangGraph Studio/API 模式下，不需要手动设置 checkpointer
# 持久化由平台自动处理
graph = builder.compile(interrupt_before=["human_feedback"])