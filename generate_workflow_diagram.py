from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from typing import List, Dict, Any, Annotated
import operator

@dataclass
class State:
    """State class for the RAG workflow"""
    messages: Annotated[List[Dict[str, str]], operator.add] = field(default_factory=list)
    current_question: str = ""
    instructions: Annotated[List[str], operator.add] = field(default_factory=list)
    context: str = ""
    calculations: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

def create_workflow_diagram():
    # Create the workflow graph
    workflow = StateGraph(State)
    
    # Add nodes with dummy functions (just for visualization)
    workflow.add_node("break_question", lambda x: x)
    workflow.add_node("retrieve_context", lambda x: x)
    workflow.add_node("calculate", lambda x: x)
    workflow.add_node("summarize", lambda x: x)
    workflow.add_node("update_buffer", lambda x: x)
    
    # Define edges
    workflow.add_edge("break_question", "retrieve_context")
    workflow.add_edge("retrieve_context", "calculate")
    workflow.add_edge("calculate", "summarize")
    workflow.add_edge("summarize", "update_buffer")
    workflow.add_edge("update_buffer", END)
    
    # Set entry point
    workflow.set_entry_point("break_question")
    
    # Compile and get the graph
    compiled = workflow.compile()
    g = compiled.get_graph()
    
    # Generate PNG
    with open("workflow_diagram.png", "wb") as f:
        f.write(g.draw_mermaid_png())
    print("✅ Workflow diagram saved as 'workflow_diagram.png'")

if __name__ == "__main__":
    create_workflow_diagram() 