"""Cross-encoder reranking using Cohere Rerank API for improved precision."""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

import cohere
from langchain_core.documents import Document

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    document: Document
    relevance_score: float
    original_index: int


class CohereReranker:
    """Cross-encoder reranker using Cohere Rerank API."""
    
    def __init__(self, model: str = "rerank-v3.5"):
        """Initialize Cohere reranker.
        
        Args:
            model: Cohere rerank model to use (rerank-v3.5 is the latest)
        """
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")
        
        self.model = model
        # Use ClientV2 as per official docs
        self.client = cohere.ClientV2(api_key=self.cohere_api_key)
        
        logger.info(f"✅ Cohere Reranker initialized with model: {model}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_n: int = 4
    ) -> Tuple[List[RerankResult], Dict[str, float]]:
        """Rerank documents using Cohere Rerank API.
        
        Args:
            query: Search query
            documents: List of LangChain Document objects
            top_n: Number of top documents to return
            
        Returns:
            Tuple of (reranked_documents, rerank_scores_dict)
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return [], {}
        
        if len(documents) <= top_n:
            logger.info(f"Only {len(documents)} documents provided, less than top_n={top_n}")
            # Return all documents with dummy scores
            results = []
            scores_dict = {}
            for i, doc in enumerate(documents):
                score = 1.0 - (i * 0.1)  # Decreasing scores
                results.append(RerankResult(
                    document=doc,
                    relevance_score=score,
                    original_index=i
                ))
                # Create score key from document content or metadata
                doc_key = self._create_doc_key(doc, i)
                scores_dict[doc_key] = score
            return results, scores_dict
        
        try:
            # Prepare documents for Cohere API
            texts = []
            for doc in documents:
                # Use page_content as the text to rerank
                text = doc.page_content.strip()
                if not text:
                    text = str(doc.metadata.get('title', 'No content'))
                texts.append(text)
            
            logger.info(f"🔄 Reranking {len(documents)} documents with query: '{query[:50]}...'")
            
            # Call Cohere Rerank API using ClientV2 (official API format)
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=texts,
                top_n=top_n
            )
            
            # Process results
            reranked_results = []
            scores_dict = {}
            
            # The response structure for ClientV2 is different
            for result in response.results:
                original_index = result.index
                relevance_score = result.relevance_score
                
                if original_index < len(documents):
                    doc = documents[original_index]
                    rerank_result = RerankResult(
                        document=doc,
                        relevance_score=relevance_score,
                        original_index=original_index
                    )
                    reranked_results.append(rerank_result)
                    
                    # Create score key for logging
                    doc_key = self._create_doc_key(doc, original_index)
                    scores_dict[doc_key] = relevance_score
            
            logger.info(
                f"✅ Reranked to top {len(reranked_results)} documents. "
                f"Score range: {min(r.relevance_score for r in reranked_results):.3f} - "
                f"{max(r.relevance_score for r in reranked_results):.3f}"
            )
            
            return reranked_results, scores_dict
            
        except Exception as e:
            logger.error(f"❌ Cohere reranking failed: {e}")
            # Fallback: return original order with dummy scores
            fallback_results = []
            fallback_scores = {}
            
            for i, doc in enumerate(documents[:top_n]):
                score = 1.0 - (i * 0.1)  # Decreasing scores
                fallback_results.append(RerankResult(
                    document=doc,
                    relevance_score=score,
                    original_index=i
                ))
                doc_key = self._create_doc_key(doc, i)
                fallback_scores[doc_key] = score
            
            logger.warning(f"Using fallback ranking for {len(fallback_results)} documents")
            return fallback_results, fallback_scores
    
    def _create_doc_key(self, doc: Document, index: int) -> str:
        """Create a unique key for document scoring."""
        # Try to use URL from metadata
        if 'source' in doc.metadata:
            source = doc.metadata['source']
            if isinstance(source, str) and source.startswith('http'):
                return source
        
        # Fallback to content hash or index
        content_preview = doc.page_content[:50].replace(' ', '_')
        return f"doc_{index}_{hash(content_preview) % 10000}"


def rerank(query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
    """Main reranking function that returns reranked documents.
    
    Args:
        query: Search query
        documents: List of Document objects to rerank
        top_n: Number of top documents to return
        
    Returns:
        List of reranked Document objects (top_n most relevant)
    """
    reranker = CohereReranker()
    results, _ = reranker.rerank(query, documents, top_n)
    return [result.document for result in results]


def rerank_with_scores(
    query: str, 
    documents: List[Document], 
    top_n: int = 4
) -> Tuple[List[Document], Dict[str, float]]:
    """Rerank documents and return both documents and scores.
    
    Args:
        query: Search query
        documents: List of Document objects to rerank
        top_n: Number of top documents to return
        
    Returns:
        Tuple of (reranked_documents, rerank_scores_dict)
    """
    reranker = CohereReranker()
    results, scores = reranker.rerank(query, documents, top_n)
    documents_reranked = [result.document for result in results]
    return documents_reranked, scores


# Convenience function for backwards compatibility
def cohere_rerank(query: str, docs: List[Document], top_n: int = 4) -> List[Document]:
    """Backwards compatible function name."""
    return rerank(query, docs, top_n)


if __name__ == "__main__":
    # Test the reranker with official API format
    from langchain_core.documents import Document
    
    print("🧪 Testing Cohere Reranker with official API format")
    print("=" * 60)
    
    # Test 1: Official example adaptation
    print("\n📝 Test 1: Official Cohere example adaptation")
    docs_official = [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word.",
        "Washington, D.C. is the capital of the United States. It is a federal district.",
        "Capital punishment has existed in the United States since before the United States was a country.",
    ]
    
    # Convert to LangChain Documents
    test_docs_official = [
        Document(
            page_content=content,
            metadata={"source": f"https://example.com/doc{i}", "title": f"Document {i}"}
        )
        for i, content in enumerate(docs_official)
    ]
    
    query_official = "What is the capital of the United States?"
    
    try:
        print(f"Query: {query_official}")
        reranked, scores = rerank_with_scores(query_official, test_docs_official, top_n=3)
        print(f"✅ Official test successful! Got {len(reranked)} documents")
        
        for i, doc in enumerate(reranked):
            content_preview = doc.page_content[:60] + "..." if len(doc.page_content) > 60 else doc.page_content
            print(f"  {i+1}. {content_preview}")
        
        print(f"Scores: {scores}")
        
    except Exception as e:
        print(f"❌ Official test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Zibtek example
    print(f"\n📝 Test 2: Zibtek example")
    test_docs_zibtek = [
        Document(
            page_content="Zibtek provides custom software development services for businesses.",
            metadata={"source": "https://zibtek.com/services", "title": "Services"}
        ),
        Document(
            page_content="We offer web development and mobile app development solutions.",
            metadata={"source": "https://zibtek.com/web-dev", "title": "Web Development"}
        ),
        Document(
            page_content="Our team specializes in PHP, Python, and JavaScript technologies.",
            metadata={"source": "https://zibtek.com/tech", "title": "Technologies"}
        ),
    ]
    
    query_zibtek = "What programming languages does Zibtek use?"
    
    try:
        print(f"Query: {query_zibtek}")
        reranked, scores = rerank_with_scores(query_zibtek, test_docs_zibtek, top_n=2)
        print(f"✅ Zibtek test successful! Got {len(reranked)} documents")
        
        for i, doc in enumerate(reranked):
            print(f"  {i+1}. {doc.metadata.get('title', 'No title')}: {doc.page_content[:50]}...")
        
        print(f"Scores: {scores}")
        
    except Exception as e:
        print(f"❌ Zibtek test failed: {e}")
        import traceback
        traceback.print_exc()