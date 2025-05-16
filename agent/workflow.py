from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from utils.model_loaders import ModelLoader 
from toolkit.tools import *


class State(TypedDict):
    messages: Annotated[list, add_messages]


class GraphBuilder:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.tools = [retriever_tool, tavily_tool, financial_tools] 
        llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.llm_with_tools = llm_with_tools
        self.graph = None
    
    def _chatbot_node(self, state: State):
        return {"messages": [self.llm_with_tools.invoke(state['messages'])]} 
    
    def build(self):
        graph_builder = StateGraph(State) 
        
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", ToolNode(self.tools))
        
        graph_builder.add_conditional_edges("chatbot",
                                            tools_condition
                                            )
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("tools", "chatbot")
        
        self.graph = graph_builder.compile()
    
    def get_graph(self):
        if self.graph is None:
            raise ValueError("Graph not built. Call build() first.")
        return self.graph