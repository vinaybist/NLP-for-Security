from typing import Dict, List, Tuple, Optional, Any, TypedDict, Annotated
from enum import Enum
import json
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import JsonCheckpointManager
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define state structure for the graph
class AnalysisState(TypedDict):
    security_control: str
    control_objective_analysis: Optional[Dict]
    statement_analysis: Optional[Dict]
    policy_standard_analysis: Optional[Dict]
    combined_analysis: Optional[Dict]
    existing_guidance: Optional[Dict]
    verification_result: Optional[bool]
    implementation_requirements: Optional[Dict]
    refinement_needed: Optional[bool]
    existing_alignment: Optional[bool]
    final_output: Optional[Dict]
    rag_results: Optional[Dict]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define node for security control input
def process_security_control(state: AnalysisState) -> AnalysisState:
    """Process the security control input and prepare it for analysis."""
    # In a real implementation, this might include preprocessing, validation, etc.
    print(f"Processing security control: {state['security_control'][:100]}...")
    return state

# Define the three parallel analyzer nodes
def control_objective_analyzer(state: AnalysisState) -> AnalysisState:
    """Analyze security control against control objectives."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a security control objective analyzer. Analyze the given security control text and extract control objectives."),
        HumanMessage(content="Analyze the following security control:\n{security_control}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"security_control": state["security_control"]})
    
    print("Control objective analysis complete")
    return {"control_objective_analysis": result}

def statement_analyzer(state: AnalysisState) -> AnalysisState:
    """Analyze security control statements."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a security statement analyzer. Parse and analyze statements in the security control."),
        HumanMessage(content="Analyze the statements in the following security control:\n{security_control}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"security_control": state["security_control"]})
    
    print("Statement analysis complete")
    return {"statement_analysis": result}

def policy_standard_analyzer(state: AnalysisState) -> AnalysisState:
    """Analyze security control against policies and standards."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a security policy and standard analyzer. Identify relevant policies and standards applicable to the given control."),
        HumanMessage(content="Identify policies and standards for the following security control:\n{security_control}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"security_control": state["security_control"]})
    
    print("Policy/standard analysis complete")
    return {"policy_standard_analysis": result}

# RAG pipeline components
def traverse_links(state: AnalysisState) -> AnalysisState:
    """Traverse links to find related content."""
    # In a real implementation, this would connect to a knowledge graph or database
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a link traversal agent. Identify key terms and concepts to search for related content."),
        HumanMessage(content="Identify searchable terms in the security control:\n{security_control}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    search_terms = chain.invoke({"security_control": state["security_control"]})
    
    print(f"Identified search terms: {json.dumps(search_terms, indent=2)}")
    return {"rag_results": {"search_terms": search_terms}}

def get_matching_content(state: AnalysisState) -> AnalysisState:
    """Retrieve top K matching content related to input."""
    # In a real implementation, this would query a vector database
    search_terms = state["rag_results"]["search_terms"]
    
    # Simulate retrieval results
    mock_results = {
        "retrieved_items": [
            {"id": 1, "content": "Sample related guidance about security implementation", "score": 0.92},
            {"id": 2, "content": "Policy reference related to the security control", "score": 0.87}
        ]
    }
    
    updated_rag_results = state["rag_results"].copy()
    updated_rag_results["retrieved_items"] = mock_results["retrieved_items"]
    
    print("Retrieved matching content")
    return {"rag_results": updated_rag_results}

def combine_analysis(state: AnalysisState) -> AnalysisState:
    """Combine the results from all three analyzers and RAG pipeline."""
    # Extract results from all analyzers
    control_results = state["control_objective_analysis"]
    statement_results = state["statement_analysis"]
    policy_results = state["policy_standard_analysis"]
    rag_results = state["rag_results"]
    
    # Create combined analysis
    combined = {
        "control_objectives": control_results,
        "statements": statement_results,
        "policies_standards": policy_results,
        "related_content": rag_results.get("retrieved_items", [])
    }
    
    print("Combined analysis complete")
    return {"combined_analysis": combined}

def fetch_existing_guidance(state: AnalysisState) -> AnalysisState:
    """Fetch existing guidance/verification from system."""
    # In a real implementation, this would query existing systems
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a guidance retrieval agent. Based on the analysis, identify relevant existing guidance."),
        HumanMessage(content="Find relevant guidance based on this analysis:\n{combined_analysis}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    guidance = chain.invoke({"combined_analysis": state["combined_analysis"]})
    
    print("Fetched existing guidance")
    return {"existing_guidance": guidance}

def verify_alignment(state: AnalysisState) -> AnalysisState:
    """Verify if Implementation Requirements are aligned with objectives and policies."""
    # Create implementation requirements based on the combined analysis and guidance
    combined = state["combined_analysis"]
    guidance = state["existing_guidance"]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a verification agent. Determine if implementation requirements align with objectives and policies."),
        HumanMessage(content="""
        Based on the following information, verify alignment:
        Combined Analysis: {combined_analysis}
        Existing Guidance: {existing_guidance}
        
        Return a JSON with:
        - aligned: true/false
        - requirements: the implementation requirements
        - reasons: reasons for alignment or misalignment
        """)
    ])
    
    chain = prompt | llm | JsonOutputParser()
    verification = chain.invoke({
        "combined_analysis": combined,
        "existing_guidance": guidance
    })
    
    # Extract verification results
    aligned = verification.get("aligned", False)
    requirements = verification.get("requirements", {})
    
    print(f"Verification complete: aligned={aligned}")
    return {
        "verification_result": aligned,
        "implementation_requirements": requirements,
        "refinement_needed": not aligned
    }

def refine_requirements(state: AnalysisState) -> AnalysisState:
    """Refine implementation requirements if not aligned."""
    current_requirements = state["implementation_requirements"]
    combined_analysis = state["combined_analysis"]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a requirement refinement agent. Refine implementation requirements to align with objectives and policies."),
        HumanMessage(content="""
        The current implementation requirements are not aligned.
        Requirements: {requirements}
        Analysis: {analysis}
        
        Refine the requirements to ensure alignment.
        """)
    ])
    
    chain = prompt | llm | JsonOutputParser()
    refined = chain.invoke({
        "requirements": current_requirements,
        "analysis": combined_analysis
    })
    
    print("Requirements refined")
    return {"implementation_requirements": refined}

def check_existing_alignment(state: AnalysisState) -> AnalysisState:
    """Check if refined requirements align with existing requirements."""
    # In a real implementation, this would compare against existing requirements database
    refined_requirements = state["implementation_requirements"]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an alignment checker. Determine if refined requirements align with existing requirements."),
        HumanMessage(content="Check if these refined requirements align with existing systems:\n{requirements}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"requirements": refined_requirements})
    
    # Parse the result to determine alignment
    aligned = "yes" in result.content.lower() or "align" in result.content.lower()
    
    print(f"Existing alignment check complete: aligned={aligned}")
    return {"existing_alignment": aligned}

def generate_final_output(state: AnalysisState) -> AnalysisState:
    """Generate final output with implementation requirements."""
    requirements = state["implementation_requirements"]
    
    final_output = {
        "implementation_requirements": requirements,
        "control_statements": [
            f"Requirement {i+1}: {req}" for i, req in enumerate(requirements.get("requirements", []))
        ],
        "metadata": {
            "aligned_with_objectives": state.get("verification_result", False),
            "aligned_with_existing": state.get("existing_alignment", False),
            "source_control": state.get("security_control", "")[:100] + "..."
        }
    }
    
    print("Final output generated")
    return {"final_output": final_output}

# Decision functions
def should_refine(state: AnalysisState) -> str:
    """Determine if refinement is needed."""
    return "refine" if state.get("refinement_needed", False) else "final"

def check_alignment(state: AnalysisState) -> str:
    """Check if aligned with existing requirements."""
    return "aligned" if state.get("existing_alignment", False) else "not_aligned"

# Create the graph
workflow = StateGraph(AnalysisState)

# Add nodes
workflow.add_node("security_control", process_security_control)
workflow.add_node("control_objective_analyzer", control_objective_analyzer)
workflow.add_node("statement_analyzer", statement_analyzer)
workflow.add_node("policy_standard_analyzer", policy_standard_analyzer)
workflow.add_node("traverse_links", traverse_links)
workflow.add_node("get_matching_content", get_matching_content)
workflow.add_node("combine_analysis", combine_analysis)
workflow.add_node("fetch_guidance", fetch_existing_guidance)
workflow.add_node("verify_alignment", verify_alignment)
workflow.add_node("refine_requirements", refine_requirements)
workflow.add_node("check_existing_alignment", check_existing_alignment)
workflow.add_node("generate_final_output", generate_final_output)

# Connect nodes
workflow.add_edge("security_control", "control_objective_analyzer")
workflow.add_edge("security_control", "statement_analyzer")
workflow.add_edge("security_control", "policy_standard_analyzer")
workflow.add_edge("security_control", "traverse_links")

workflow.add_edge("traverse_links", "get_matching_content")
workflow.add_edge("get_matching_content", "combine_analysis")
workflow.add_edge("control_objective_analyzer", "combine_analysis")
workflow.add_edge("statement_analyzer", "combine_analysis")
workflow.add_edge("policy_standard_analyzer", "combine_analysis")

workflow.add_edge("combine_analysis", "fetch_guidance")
workflow.add_edge("fetch_guidance", "verify_alignment")

# Add conditional edges
workflow.add_conditional_edges(
    "verify_alignment",
    should_refine,
    {
        "refine": "refine_requirements",
        "final": "generate_final_output"
    }
)

workflow.add_edge("refine_requirements", "check_existing_alignment")

workflow.add_conditional_edges(
    "check_existing_alignment",
    check_alignment,
    {
        "aligned": "generate_final_output",
        "not_aligned": "refine_requirements"
    }
)

workflow.add_edge("generate_final_output", END)

# Compile the graph
app = workflow.compile()

# Function to run the workflow
def analyze_security_control(security_control: str) -> Dict:
    """Run the security control analysis workflow."""
    # Initialize state
    initial_state = AnalysisState(
        security_control=security_control,
        control_objective_analysis=None,
        statement_analysis=None,
        policy_standard_analysis=None,
        combined_analysis=None,
        existing_guidance=None,
        verification_result=None,
        implementation_requirements=None,
        refinement_needed=None,
        existing_alignment=None,
        final_output=None,
        rag_results=None
    )
    
    # Create checkpoint manager (optional, for persistence)
    # checkpoint = JsonCheckpointManager("./checkpoints/")
    
    # Run the workflow
    # result = app.invoke(initial_state, config={"checkpoint": checkpoint})
    result = app.invoke(initial_state)
    
    # Return the final output
    return result["final_output"]

# Example usage
if __name__ == "__main__":
    sample_control = """
    Access Control Policy:
    All systems must implement role-based access control (RBAC) with principle of least privilege.
    Access to critical systems requires multi-factor authentication.
    All access attempts must be logged and monitored.
    Accounts must be reviewed quarterly and unused accounts disabled after 90 days of inactivity.
    """
    
    result = analyze_security_control(sample_control)
    print("\nFinal Output:")
    print(json.dumps(result, indent=2))
