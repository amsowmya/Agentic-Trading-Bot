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
        pass 
    
    def _chatbot_node(self, state: State):
        pass 
    
    def build():
        pass 
    
    def get_graph(self):
        pass