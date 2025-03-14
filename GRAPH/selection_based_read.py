"""
PDF Section Chunking with DocLing for RAG Applications
"""

import os
from typing import List, Dict, Optional, Union, Tuple

from docling import Document
from docling.document import DocumentTree
from langchain_core.documents import Document as LCDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocLingPDFChunker:
    """
    A class for extracting and chunking PDF content from specific sections using DocLing.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
    ):
        """
        Initialize the DocLingPDFChunker.
        
        Args:
            chunk_size: The target size of each text chunk
            chunk_overlap: The overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> Document:
        """
        Load a PDF file using DocLing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocLing Document object
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Load the document with DocLing
        doc = Document.from_pdf(pdf_path)
        
        # Process document to identify sections
        doc.process()
        
        return doc
    
    def get_section_tree(self, doc: Document) -> Dict[str, any]:
        """
        Get the document's section tree structure.
        
        Args:
            doc: DocLing Document object
            
        Returns:
            Dictionary containing the section structure
        """
        if not hasattr(doc, 'tree') or doc.tree is None:
            raise ValueError("Document does not have a processed section tree. Call doc.process() first.")
        
        def _convert_tree_to_dict(node: DocumentTree) -> Dict:
            result = {
                "heading": node.heading if hasattr(node, 'heading') else None,
                "level": node.level if hasattr(node, 'level') else None,
                "children": []
            }
            
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    result["children"].append(_convert_tree_to_dict(child))
            
            return result
        
        return _convert_tree_to_dict(doc.tree)
    
    def list_available_sections(self, doc: Document) -> List[str]:
        """
        List all available sections in the document.
        
        Args:
            doc: DocLing Document object
            
        Returns:
            List of section headings
        """
        sections = []
        
        def _collect_headings(node: DocumentTree) -> None:
            if hasattr(node, 'heading') and node.heading:
                sections.append(node.heading)
            
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    _collect_headings(child)
        
        _collect_headings(doc.tree)
        return sections
    
    def _find_section_node(
        self, 
        node: DocumentTree, 
        section_name: str
    ) -> Optional[DocumentTree]:
        """
        Find a section node by name in the document tree.
        
        Args:
            node: Current document tree node
            section_name: Name of the section to find
            
        Returns:
            DocumentTree node if found, None otherwise
        """
        if hasattr(node, 'heading') and node.heading == section_name:
            return node
        
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                result = self._find_section_node(child, section_name)
                if result:
                    return result
        
        return None
    
    def get_section_text(
        self, 
        doc: Document, 
        section_name: str
    ) -> Tuple[str, Dict]:
        """
        Extract text from a specific section.
        
        Args:
            doc: DocLing Document object
            section_name: Name of the section to extract
            
        Returns:
            Tuple of (section_text, section_metadata)
        """
        available_sections = self.list_available_sections(doc)
        
        if section_name not in available_sections:
            raise ValueError(
                f"Section '{section_name}' not found. Available sections: {available_sections}"
            )
        
        section_node = self._find_section_node(doc.tree, section_name)
        
        if not section_node:
            raise ValueError(f"Section node for '{section_name}' could not be found")
        
        # Extract text from this section
        section_text = section_node.get_text()
        
        # Create metadata
        metadata = {
            "source": doc.src if hasattr(doc, 'src') else "Unknown",
            "section": section_name,
            "level": section_node.level if hasattr(section_node, 'level') else None,
            "page_range": section_node.page_range if hasattr(section_node, 'page_range') else None
        }
        
        return section_text, metadata
    
    def chunk_section(
        self, 
        pdf_path: str, 
        section_name: str
    ) -> List[LCDocument]:
        """
        Extract and chunk text from a specific section of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            section_name: Name of the section to extract and chunk
            
        Returns:
            List of LangChain Document objects with chunked text
        """
        doc = self.load_pdf(pdf_path)
        
        # Show available sections to help with debugging
        available_sections = self.list_available_sections(doc)
        print(f"Available sections: {available_sections}")
        
        section_text, metadata = self.get_section_text(doc, section_name)
        
        # Create chunks
        chunks = self.text_splitter.create_documents(
            texts=[section_text],
            metadatas=[metadata]
        )
        
        return chunks
    
    def chunk_multiple_sections(
        self, 
        pdf_path: str, 
        section_names: List[str]
    ) -> Dict[str, List[LCDocument]]:
        """
        Extract and chunk text from multiple sections of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            section_names: List of section names to extract and chunk
            
        Returns:
            Dictionary mapping section names to lists of Document objects
        """
        doc = self.load_pdf(pdf_path)
        available_sections = self.list_available_sections(doc)
        
        # Validate section names
        for section in section_names:
            if section not in available_sections:
                raise ValueError(
                    f"Section '{section}' not found. Available sections: {available_sections}"
                )
        
        results = {}
        for section in section_names:
            section_text, metadata = self.get_section_text(doc, section)
            
            # Create chunks
            chunks = self.text_splitter.create_documents(
                texts=[section_text],
                metadatas=[metadata]
            )
            
            results[section] = chunks
        
        return results


# Example usage
if __name__ == "__main__":
    chunker = DocLingPDFChunker(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Example 1: List available sections in a PDF
    pdf_path = "example.pdf"
    doc = chunker.load_pdf(pdf_path)
    sections = chunker.list_available_sections(doc)
    print(f"Available sections in {pdf_path}:")
    for i, section in enumerate(sections):
        print(f"  {i+1}. {section}")
    
    # Example 2: Chunk a single section
    try:
        section_name = "Introduction"  # Update with an actual section from your PDF
        chunks = chunker.chunk_section(pdf_path, section_name)
        print(f"\nCreated {len(chunks)} chunks from section '{section_name}'")
        
        # Preview first chunk
        if chunks:
            print("\nFirst chunk preview:")
            print(f"Text: {chunks[0].page_content[:150]}...")
            print(f"Metadata: {chunks[0].metadata}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Chunk multiple sections for RAG
    try:
        # Update with actual sections from your PDF
        section_names = ["Abstract", "Introduction", "Methodology"] 
        section_chunks = chunker.chunk_multiple_sections(pdf_path, section_names)
        
        for section, chunks in section_chunks.items():
            print(f"\nSection '{section}': {len(chunks)} chunks")
            if chunks:
                print(f"First chunk preview: {chunks[0].page_content[:100]}...")
                print(f"Metadata: {chunks[0].metadata}")
    except Exception as e:
        print(f"Error: {e}")
