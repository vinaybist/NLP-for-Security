import os
import yaml
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Union, Literal
from dotenv import load_dotenv
import json
import hashlib
import lancedb
from docling import Doc
from docling.readers import PDFReader
from docling.chunkers import HybridChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Define output structures
class RequirementSection(BaseModel):
    statements_activities: List[str] = Field(description="List of statement or activities requirements")
    verification: List[str] = Field(description="List of verification requirements")
    guidance: List[str] = Field(description="List of guidance requirements")

class ImplementationRequirement(BaseModel):
    name: str = Field(description="Name of the control or requirement")
    id: str = Field(description="Identifier for the requirement")
    sections: RequirementSection = Field(description="The three sections of implementation requirements")
    
class ReflectionResult(BaseModel):
    satisfied: bool = Field(description="Whether the requirements are satisfactory")
    reasoning: str = Field(description="Reasoning behind the satisfaction decision")
    refinement_suggestions: Optional[List[str]] = Field(default=None, description="Suggestions for refinement if not satisfied")

# State definition for LangGraph
class WorkflowState(TypedDict):
    security_control: str
    standard_requirements: Optional[Dict]
    policy_requirements: Optional[Dict]
    consolidated_requirements: Optional[Dict]
    existing_requirements: Optional[Dict]
    reflection_result: Optional[Dict]
    refinement_count: int
    final_output: Optional[Dict]


# Document processor using IBM's Docling
class DocumentProcessor:
    """Process documents using IBM's Docling for PDF extraction and hybrid chunking"""
    
    def __init__(self, data_dir="./data"):
        """Initialize the document processor"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.pdf_reader = PDFReader()
        
    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """Load and extract text from PDF documents using IBM's Docling"""
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                # Use IBM's Docling to extract text and structure from PDF
                doc = Doc()
                doc = self.pdf_reader.read(pdf_path, doc)
                
                # Extract the text content
                # Create a Document object for each page to maintain structure
                for page_num, page in enumerate(doc.pages):
                    langchain_doc = Document(
                        page_content=page.text,
                        metadata={
                            "source": pdf_path, 
                            "type": "pdf",
                            "page_number": page_num + 1,
                            "total_pages": len(doc.pages)
                        }
                    )
                    documents.append(langchain_doc)
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")
        
        return documents
    
    def chunk_documents(self, documents: List[Document], 
                        chunk_size=1000, 
                        chunk_overlap=200) -> List[Document]:
        """Chunk documents using Docling's HybridChunker"""
        chunker = HybridChunker(
            semantic_chunker_params={"ideal_chunk_size": chunk_size, "overlap": chunk_overlap},
            layout_chunker_params={"min_chunk_size": chunk_size // 2}
        )
        
        # Process each document with Docling for intelligent chunking
        chunked_documents = []
        
        for doc in documents:
            # Create a Docling Doc from the langchain Document
            docling_doc = Doc()
            docling_doc.text = doc.page_content
            
            # Apply the hybrid chunker
            chunked_doc = chunker.chunk(docling_doc)
            
            # Convert the chunks back to langchain Documents
            for i, chunk in enumerate(chunked_doc.chunks):
                chunk_doc = Document(
                    page_content=chunk.text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "chunk_type": chunk.chunk_type if hasattr(chunk, "chunk_type") else "hybrid"
                    }
                )
                chunked_documents.append(chunk_doc)
        
        return chunked_documents


# RAG components using LanceDB
class LanceDBRAG:
    """RAG tool using LanceDB for vector storage"""
    
    def __init__(self, document_type: str, documents: List[str] = None, pdf_paths: List[str] = None, 
                 db_path="./lancedb", chunk_size=1000, chunk_overlap=200):
        """
        Initialize the LanceDB RAG tool
        
        Args:
            document_type: Type of documents ("standard" or "policy")
            documents: Optional list of text documents
            pdf_paths: Optional list of paths to PDF documents
            db_path: Path to LanceDB database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.document_type = document_type
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_name = f"{document_type}_vectors"
        
        # Create LanceDB connection
        self.db = lancedb.connect(db_path)
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor()
        
        # Create or get the vector table
        if documents or pdf_paths:
            self._create_or_update_table(documents, pdf_paths)
    
    def _create_or_update_table(self, documents: List[str] = None, pdf_paths: List[str] = None):
        """Create or update the vector table"""
        # Check if table already exists
        if self.table_name in self.db.table_names():
            print(f"Table {self.table_name} already exists. Using existing table.")
            self.table = self.db.open_table(self.table_name)
            return
        
        # Process documents if provided
        all_chunks = []
        
        # Process text documents
        if documents:
            text_docs = [Document(page_content=doc, metadata={"source": f"{self.document_type}_{i}", "type": "text"}) 
                         for i, doc in enumerate(documents)]
            text_chunks = self.doc_processor.chunk_documents(text_docs, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(text_chunks)
        
        # Process PDF documents
        if pdf_paths:
            pdf_docs = self.doc_processor.load_pdf_documents(pdf_paths)
            pdf_chunks = self.doc_processor.chunk_documents(pdf_docs, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(pdf_chunks)
        
        # If no chunks were created, return
        if not all_chunks:
            print(f"No documents provided for {self.table_name}. Table not created.")
            return
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector data for LanceDB
        vector_data = []
        for i, chunk in enumerate(all_chunks):
            vector = embeddings.embed_query(chunk.page_content)
            vector_data.append({
                "id": str(i),
                "vector": vector,
                "text": chunk.page_content,
                "metadata": json.dumps(chunk.metadata)
            })
        
        # Create the table
        self.table = self.db.create_table(
            self.table_name,
            data=vector_data,
            mode="overwrite"
        )
        print(f"Created vector table {self.table_name} with {len(vector_data)} entries.")
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top K relevant documents for the query"""
        # Create embeddings for the query
        embeddings = OpenAIEmbeddings()
        query_vector = embeddings.embed_query(query)
        
        # Search the vector table
        results = self.table.search(query_vector).limit(k).to_pandas()
        
        # Extract the text from the results
        return results["text"].tolist()


# Node definitions for LangGraph
def standard_analyzer(state: WorkflowState, standard_documents: List[str] = None, 
                     standard_pdfs: List[str] = None, llm=None) -> WorkflowState:
    """Standard Analyzer node"""
    llm = llm or ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    
    # Create RAG tool
    rag_tool = LanceDBRAG("standard", documents=standard_documents, pdf_paths=standard_pdfs)
    
    # Retrieve relevant context
    context = rag_tool.retrieve(state["security_control"])
    
    # Create standard analyzer prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Standard Analyzer agent. Your task is to analyze the given security control 
        and generate implementation requirements based on standard considerations.
        
        Generate implementation requirements that are:
        1. Clear and actionable
        2. Aligned with standard best practices
        3. Structured into three categories: Statement/Activities, Verification, and Guidance
        
        Output should follow the format:
        {format_instructions}
        """),
        ("human", """
        Security Control: {security_control}
        
        Relevant Context:
        {context}
        
        Based on the security control and context, generate implementation requirements.
        """)
    ])
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=ReflectionResult)
    
    # Run the chain
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "consolidated_requirements": json.dumps(state["consolidated_requirements"], indent=2),
        "existing_requirements": json.dumps(state["existing_requirements"], indent=2)
    })
    
    # Update state
    return {**state, "reflection_result": result.dict()}

def refinement_agent(state: WorkflowState, llm=None) -> WorkflowState:
    """Refinement Agent node"""
    llm = llm or ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    
    # Create refinement prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Refinement Agent. Your task is to refine the consolidated requirements based on 
        reflection feedback and comparison with existing requirements. Consider:
        
        1. Address all refinement suggestions
        2. Ensure no important information is lost from existing requirements
        3. Make the requirements clearer, more actionable, and comprehensive
        
        Output the refined requirements as:
        {format_instructions}
        """),
        ("human", """
        Consolidated Requirements:
        {consolidated_requirements}
        
        Existing Requirements:
        {existing_requirements}
        
        Refinement Suggestions:
        {refinement_suggestions}
        
        Reflection Reasoning:
        {reflection_reasoning}
        
        Please refine the consolidated requirements based on the feedback.
        """)
    ])
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=ImplementationRequirement)
    
    # Run the chain
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "consolidated_requirements": json.dumps(state["consolidated_requirements"], indent=2),
        "existing_requirements": json.dumps(state["existing_requirements"], indent=2),
        "refinement_suggestions": json.dumps(state["reflection_result"].get("refinement_suggestions", []), indent=2),
        "reflection_reasoning": state["reflection_result"].get("reasoning", "")
    })
    
    # Update state with refined requirements and increment refinement count
    return {
        **state, 
        "consolidated_requirements": result.dict(),
        "refinement_count": state["refinement_count"] + 1
    }

def finalize_output(state: WorkflowState) -> WorkflowState:
    """Finalize output node"""
    # Set the final output to the consolidated requirements
    return {**state, "final_output": state["consolidated_requirements"]}

def route_reflection(state: WorkflowState) -> Union[Literal["refine"], Literal["finalize"]]:
    """Determine whether to refine or finalize based on reflection"""
    reflection_result = state.get("reflection_result", {})
    refinement_count = state.get("refinement_count", 0)
    
    # If reflection is satisfied or we've reached max refinements, finalize
    if reflection_result.get("satisfied", True) or refinement_count >= 3:
        return "finalize"
    else:
        return "refine"


class MultiAgentWorkflowGraph:
    """LangGraph implementation of the multi-agent workflow for generating implementation requirements"""
    
    def __init__(self, 
                 standard_documents: List[str] = None,
                 policy_documents: List[str] = None,
                 standard_pdfs: List[str] = None,
                 policy_pdfs: List[str] = None,
                 existing_requirements: Optional[Dict[str, Any]] = None,
                 llm=None,
                 lancedb_path="./lancedb"):
        """
        Initialize the workflow graph
        
        Args:
            standard_documents: Optional list of standard documents as text
            policy_documents: Optional list of policy documents as text
            standard_pdfs: Optional list of paths to standard PDF documents
            policy_pdfs: Optional list of paths to policy PDF documents
            existing_requirements: Optional existing implementation requirements
            llm: Optional language model for the agents
            lancedb_path: Path to store LanceDB files
        """
        self.standard_documents = standard_documents or []
        self.policy_documents = policy_documents or []
        self.standard_pdfs = standard_pdfs or []
        self.policy_pdfs = policy_pdfs or []
        self.existing_requirements = existing_requirements
        self.llm = llm or ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
        self.lancedb_path = lancedb_path
        
        # Ensure we have valid documents
        if not (self.standard_documents or self.standard_pdfs):
            print("Warning: No standard documents or PDFs provided.")
        
        if not (self.policy_documents or self.policy_pdfs):
            print("Warning: No policy documents or PDFs provided.")
        
        # Initialize LanceDB
        os.makedirs(lancedb_path, exist_ok=True)
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
    def _build_graph(self):
        """Build the LangGraph workflow"""
        # Define the workflow state
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("standard_analyzer", lambda state: standard_analyzer(
            state, 
            standard_documents=self.standard_documents,
            standard_pdfs=self.standard_pdfs,
            llm=self.llm
        ))
        
        workflow.add_node("policy_analyzer", lambda state: policy_analyzer(
            state, 
            policy_documents=self.policy_documents,
            policy_pdfs=self.policy_pdfs,
            llm=self.llm
        ))
        
        workflow.add_node("consolidation", lambda state: consolidation_agent(state, self.llm))
        workflow.add_node("reflection", lambda state: reflection_agent(state, self.llm))
        workflow.add_node("refinement", lambda state: refinement_agent(state, self.llm))
        workflow.add_node("finalize", finalize_output)
        
        # Set the entry points
        workflow.set_entry_point("standard_analyzer")
        workflow.set_entry_point("policy_analyzer")
        
        # Define edges
        # Start with parallel analyzers
        workflow.add_edge("standard_analyzer", "consolidation")
        workflow.add_edge("policy_analyzer", "consolidation")
        
        # After consolidation, perform reflection
        workflow.add_edge("consolidation", "reflection")
        
        # After reflection, either refine or finalize
        workflow.add_conditional_edges(
            "reflection",
            route_reflection,
            {
                "refine": "refinement",
                "finalize": "finalize"
            }
        )
        
        # After refinement, go back to reflection
        workflow.add_edge("refinement", "reflection")
        
        # Finalize is an end state
        workflow.add_edge("finalize", END)
        
        return workflow
        
    async def run(self, security_control: str) -> Dict[str, Any]:
        """
        Run the workflow on a security control input
        
        Args:
            security_control: The security control text to analyze
            
        Returns:
            The final implementation requirements
        """
        # Initialize the state
        initial_state = {
            "security_control": security_control,
            "standard_requirements": None,
            "policy_requirements": None,
            "consolidated_requirements": None,
            "existing_requirements": self.existing_requirements,
            "reflection_result": None,
            "refinement_count": 0,
            "final_output": None
        }
        
        # Run the workflow
        result = await self.app.ainvoke(initial_state)
        
        # Return the final output
        return result["final_output"]
    
    def generate_yaml(self, requirements: Dict[str, Any], output_file: str) -> str:
        """
        Generate YAML file from requirements
        
        Args:
            requirements: The implementation requirements
            output_file: Path to save the YAML file
            
        Returns:
            Success message
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(requirements, f, default_flow_style=False)
        
        return f"YAML file generated: {output_file}"dantic_object=ImplementationRequirement)
    
    # Run the chain
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "security_control": state["security_control"],
        "context": "\n\n".join(context)
    })
    
    # Update state
    return {**state, "standard_requirements": result.dict()}

def policy_analyzer(state: WorkflowState, policy_documents: List[str] = None,
                   policy_pdfs: List[str] = None, llm=None) -> WorkflowState:
    """Policy Analyzer node"""
    llm = llm or ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    
    # Create RAG tool
    rag_tool = LanceDBRAG("policy", documents=policy_documents, pdf_paths=policy_pdfs)
    
    # Retrieve relevant context
    context = rag_tool.retrieve(state["security_control"])
    
    # Create policy analyzer prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Policy Analyzer agent. Your task is to analyze the given security control 
        and generate implementation requirements based on policy considerations.
        
        Generate implementation requirements that are:
        1. Clear and actionable
        2. Aligned with policy best practices
        3. Structured into three categories: Statement/Activities, Verification, and Guidance
        
        Output should follow the format:
        {format_instructions}
        """),
        ("human", """
        Security Control: {security_control}
        
        Relevant Context:
        {context}
        
        Based on the security control and context, generate implementation requirements.
        """)
    ])
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=ImplementationRequirement)
    
    # Run the chain
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "security_control": state["security_control"],
        "context": "\n\n".join(context)
    })
    
    # Update state
    return {**state, "policy_requirements": result.dict()}

def consolidation_agent(state: WorkflowState, llm=None) -> WorkflowState:
    """Consolidation Agent node"""
    llm = llm or ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    
    # Create consolidation prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Consolidation Agent responsible for combining and refining implementation requirements 
        from standard and policy analyses. Your task is to:
        
        1. Combine requirements from both sources ensuring no information is lost
        2. Structure the output into three clear sections: Statement/Activities, Verification, and Guidance
        3. Compare with existing requirements (if provided)
        4. Ensure the consolidated requirements are comprehensive, clear, and actionable
        
        Output should follow the format:
        {format_instructions}
        """),
        ("human", """
        Standard Requirements:
        {standard_requirements}
        
        Policy Requirements:
        {policy_requirements}
        
        {existing_requirements_text}
        
        Please consolidate these requirements into a unified set of implementation requirements.
        """)
    ])
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=ImplementationRequirement)
    
    # Prepare existing requirements text
    existing_requirements_text = ""
    if state.get("existing_requirements"):
        existing_requirements_text = f"""
        Existing Implementation Requirements:
        {json.dumps(state['existing_requirements'], indent=2)}
        """
    
    # Run the chain
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "standard_requirements": json.dumps(state["standard_requirements"], indent=2),
        "policy_requirements": json.dumps(state["policy_requirements"], indent=2),
        "existing_requirements_text": existing_requirements_text
    })
    
    # Update state
    return {**state, "consolidated_requirements": result.dict()}

def reflection_agent(state: WorkflowState, llm=None) -> WorkflowState:
    """Reflection Agent node"""
    # Skip reflection if no existing requirements
    if not state.get("existing_requirements"):
        return {**state, 
                "reflection_result": {"satisfied": True, "reasoning": "No existing requirements to compare with."}, 
                "final_output": state["consolidated_requirements"]}
    
    llm = llm or ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    
    # Create reflection prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a Reflection Agent. Your task is to compare consolidated requirements with existing ones 
        and determine if the consolidated requirements are satisfactory. Consider:
        
        1. Are all key points from existing requirements preserved?
        2. Are the consolidated requirements more refined and clearer?
        3. Is any important information lost in the consolidation?
        
        Output your reflection as:
        {format_instructions}
        """),
        ("human", """
        Consolidated Requirements:
        {consolidated_requirements}
        
        Existing Requirements:
        {existing_requirements}
        
        Please reflect on whether the consolidated requirements are satisfactory.
        """)
    ])
    
    # Create parser
    parser = PydanticOutputParser(py
