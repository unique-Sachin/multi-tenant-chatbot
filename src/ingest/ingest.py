"""Milvus-based data ingestion pipeline for Zibtek chatbot.

This script crawls websites, extracts text content, chunks it,
generates embeddings, and stores them in Milvus with hybrid search support (dense + BM25).
"""

import os
import sys
import hashlib
import requests
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import trafilatura
from bs4 import BeautifulSoup
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

load_dotenv()


class MilvusIngestor:
    """Milvus-based ingestion pipeline with hybrid search support (dense + BM25)."""
    
    def __init__(self, domain: Optional[str] = None, partition_name: Optional[str] = None, max_pages: Optional[int] = None):
        """Initialize Milvus ingestion pipeline.
        
        Args:
            domain: Website domain to crawl
            partition_name: Milvus partition name for multi-tenant isolation (e.g., "org_zibtek")
            max_pages: Maximum pages to crawl
        """
        self.domain = domain or os.getenv("DATASET_DOMAIN", "https://www.zibtek.com")
        self.partition_name = partition_name or "_default"
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Configuration
        self.chunk_size = 1000  # tokens
        self.chunk_overlap = 200  # tokens
        self.max_pages = max_pages or 500
        self.crawl_timestamp = datetime.now().isoformat()
        
        # Import Milvus storage
        from src.storage.milvus import milvus_storage, init_milvus_collection, create_milvus_partition
        
        self.milvus_storage = milvus_storage
        
        # Initialize collection
        if not init_milvus_collection():
            raise ValueError("Failed to initialize Milvus collection")
        
        # Create partition for this tenant
        if not create_milvus_partition(self.partition_name):
            raise ValueError(f"Failed to create partition: {self.partition_name}")
        
        print(f"✅ Milvus Ingestor initialized")
        print(f"   - Domain: {self.domain}")
        print(f"   - Partition: {self.partition_name}")
        print(f"   - Chunk size: {self.chunk_size} tokens")
        print(f"   - Chunk overlap: {self.chunk_overlap} tokens")
    
    def discover_urls(self) -> List[str]:
        """Discover URLs to crawl from sitemap or homepage."""
        urls = []
        
        # Try to get sitemap first
        sitemap_urls = self._get_sitemap_urls()
        if sitemap_urls:
            print(f"✅ Found {len(sitemap_urls)} URLs from sitemap")
            urls.extend(sitemap_urls)
        else:
            print("⚠️  No sitemap found, using homepage only")
            urls.append(self.domain)
        
        # Filter and deduplicate URLs
        filtered_urls = self._filter_urls(urls)
        
        # Limit for testing
        if len(filtered_urls) > self.max_pages:
            filtered_urls = filtered_urls[:self.max_pages]
            print(f"⚠️  Limited to first {self.max_pages} pages for testing")
        
        print(f"🎯 Will crawl {len(filtered_urls)} URLs")
        return filtered_urls
    
    def _get_sitemap_urls(self) -> List[str]:
        """Extract URLs from sitemap with support for multiple patterns and index files."""
        all_urls = []
        
        # Try multiple common sitemap patterns
        sitemap_patterns = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap-index.xml",
            "/pages-sitemap.xml",
            "/post-sitemap.xml",
            "/sitemap1.xml",
            "/sitemap/sitemap.xml",
            "/wp-sitemap.xml",  # WordPress
        ]
        
        for pattern in sitemap_patterns:
            sitemap_url = urljoin(self.domain, pattern)
            urls = self._fetch_sitemap(sitemap_url)
            if urls:
                all_urls.extend(urls)
                print(f"✅ Found {len(urls)} URLs from {pattern}")
        
        # Deduplicate
        all_urls = list(dict.fromkeys(all_urls))
        
        if all_urls:
            print(f"✅ Total URLs found from all sitemaps: {len(all_urls)}")
        else:
            print("⚠️  No sitemaps found")
        
        return all_urls
    
    def _fetch_sitemap(self, sitemap_url: str) -> List[str]:
        """Fetch and parse a single sitemap, handling both sitemap indices and regular sitemaps."""
        try:
            response = requests.get(sitemap_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ZibtekBot/1.0)'
            })
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Handle namespace
            namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            
            # Check if this is a sitemap index (contains references to other sitemaps)
            sitemap_refs = root.findall('.//sitemap:sitemap', namespace)
            if not sitemap_refs:
                sitemap_refs = root.findall('.//sitemap', namespace)
            
            if sitemap_refs:
                # This is a sitemap index - fetch all referenced sitemaps
                print(f"   📑 Found sitemap index with {len(sitemap_refs)} sub-sitemaps")
                all_urls = []
                
                for sitemap_ref in sitemap_refs:
                    loc_elem = sitemap_ref.find('sitemap:loc', namespace)
                    if loc_elem is None:
                        loc_elem = sitemap_ref.find('loc')
                    
                    if loc_elem is not None and loc_elem.text:
                        sub_sitemap_url = loc_elem.text
                        print(f"   🔗 Fetching sub-sitemap: {sub_sitemap_url}")
                        sub_urls = self._fetch_sitemap(sub_sitemap_url)
                        all_urls.extend(sub_urls)
                
                return all_urls
            
            # This is a regular sitemap - extract URLs
            urls = []
            
            # Try with namespace
            for url_elem in root.findall('.//sitemap:url', namespace):
                loc_elem = url_elem.find('sitemap:loc', namespace)
                if loc_elem is not None and loc_elem.text:
                    urls.append(loc_elem.text)
            
            # Fallback: try without namespace
            if not urls:
                for url_elem in root.findall('.//url'):
                    loc_elem = url_elem.find('loc')
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text)
            
            return urls
            
        except requests.exceptions.HTTPError as e:
            # 404 is expected for non-existent sitemaps, don't spam logs
            if e.response.status_code != 404:
                print(f"   ⚠️  HTTP error fetching {sitemap_url}: {e}")
            return []
        except Exception as e:
            print(f"   ⚠️  Error fetching {sitemap_url}: {e}")
            return []
    
    def _filter_urls(self, urls: List[str]) -> List[str]:
        """Filter URLs to only include in-domain pages."""
        domain_parts = urlparse(self.domain)
        base_domain = domain_parts.netloc
        
        filtered = []
        seen = set()
        
        for url in urls:
            try:
                parsed = urlparse(url)
                
                # Only include same domain
                if parsed.netloc != base_domain:
                    continue
                
                # Skip common non-content files
                path = parsed.path.lower()
                if any(ext in path for ext in ['.pdf', '.jpg', '.png', '.css', '.js', '.xml', '.txt']):
                    continue
                
                # Normalize URL
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if normalized not in seen:
                    seen.add(normalized)
                    filtered.append(normalized)
                    
            except Exception as e:
                print(f"⚠️  Skipping invalid URL {url}: {e}")
                continue
        
        return filtered
    
    def crawl_pages(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Crawl and extract content from pages."""
        pages = []
        
        for i, url in enumerate(urls, 1):
            print(f"📄 Crawling page {i}/{len(urls)}: {url}")
            
            try:
                content_data = self._extract_page_content(url)
                if content_data:
                    pages.append(content_data)
                    print(f"   ✅ Extracted {len(content_data['text'])} characters")
                else:
                    print(f"   ❌ Failed to extract content")
                
                # Small delay to be respectful
                time.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Error crawling {url}: {e}")
                continue
        
        print(f"✅ Successfully crawled {len(pages)} pages")
        return pages
    
    def _extract_page_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract clean text content from a single page."""
        try:
            # Download page
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ZibtekBot/1.0)'
            })
            response.raise_for_status()
            
            # Try trafilatura first (best for content extraction)
            content = trafilatura.extract(
                response.text,
                include_formatting=True,
                include_links=False
            )
            
            title = ""  
            
            # Fallback to BeautifulSoup if trafilatura fails
            if not content or len(content.strip()) < 100:
                print(f"   ⚠️  Trafilatura failed, using BeautifulSoup fallback")
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else ""
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    element.decompose()
                
                # Get main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main'])
                if main_content:
                    content = main_content.get_text(separator=' ', strip=True)
                else:
                    content = soup.get_text(separator=' ', strip=True)
            
            # Extract title from trafilatura content if not found
            if not title and content:
                lines = content.split('\n')
                if lines and len(lines[0].strip()) < 100:
                    title = lines[0].strip()
            
            # Clean and validate content
            if not content or len(content.strip()) < 50:
                return None
            
            content = self._clean_text(content)
            
            return {
                'url': url,
                'title': title,
                'text': content,
                'length': len(content)
            }
            
        except Exception as e:
            print(f"   ❌ Error extracting content from {url}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common footer/header text patterns
        patterns_to_remove = [
            r'©\s*\d{4}.*?rights reserved',
            r'all rights reserved',
            r'privacy policy',
            r'terms of service',
            r'cookie policy'
        ]
        
        import re
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def chunk_content(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk page content into smaller pieces with metadata."""
        all_chunks = []
        
        for page in pages:
            print(f"📝 Chunking content from: {page['url']}")
            
            chunks = self._create_chunks_recursive(
                page['text'], 
                page['url'], 
                page['title']
            )
            
            all_chunks.extend(chunks)
            print(f"   ✅ Created {len(chunks)} chunks")
        
        print(f"✅ Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _create_chunks_recursive(self, text: str, url: str, title: str) -> List[Dict[str, Any]]:
        """Create chunks using recursive splitting with better structure awareness."""
        
        # Initialize recursive splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # tokens (will convert to chars)
            chunk_overlap=self.chunk_overlap,  # tokens (will convert to chars)
            length_function=self._count_tokens,  # Use tiktoken for accurate counting
            separators=[
                "\n\n\n",  # Triple newlines (major sections)
                "\n\n",    # Double newlines (paragraphs)
                "\n",      # Single newlines
                ". ",      # Sentences
                ", ",      # Clauses
                " ",       # Words
                ""         # Characters (last resort)
            ],
            keep_separator=True
        )
        
        # Split the text
        chunk_texts = splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = self._generate_chunk_id(url, chunk_text)
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'metadata': {
                    'url': url,
                    'title': title,
                    'section': self._extract_section(chunk_text),
                    'crawl_ts': self.crawl_timestamp,
                    'site': self.domain,
                    'partition_name': self.partition_name,
                    'chunk_index': i,
                    'token_count': self._count_tokens(chunk_text)
                }
            })
        
        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.tokenizer.encode(text))
 
    def _generate_chunk_id(self, url: str, text: str) -> str:
        """Generate deterministic ID for chunk."""
        content = f"{url}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _extract_section(self, text: str) -> str:
        """Extract the most relevant section heading from chunk text."""
        # Look for heading-like patterns
        lines = text.split('\n')
        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            if len(line) < 100 and len(line) > 5:  # Reasonable heading length
                return line
        
        # Fallback: use first 50 characters
        return text[:50] + "..." if len(text) > 50 else text
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks."""
        print(f"🔮 Generating embeddings for {len(chunks)} chunks...")
        
        # Prepare texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                response = self.openai_client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-small"
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error generating embeddings for batch: {e}")
                # Add empty embeddings as fallback
                all_embeddings.extend([[0.0] * 1536] * len(batch_texts))
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = all_embeddings[i]
        
        print(f"✅ Generated embeddings for all chunks")
        return chunks
    
    def store_in_milvus(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store chunks in Milvus with hybrid search support.
        
        Milvus will auto-generate sparse_vector from text field using BM25 function.
        
        Args:
            chunks: List of chunks with embeddings
        
        Returns:
            True if successful
        """
        if not chunks:
            print("⚠️  No chunks to store")
            return False
        
        print(f"\n{'='*60}")
        print(f"� Storing {len(chunks)} chunks in Milvus")
        print(f"   - Collection: documents")
        print(f"   - Partition: {self.partition_name}")
        print('='*60)
        
        try:
            # Prepare documents for Milvus
            documents = []
            for chunk in chunks:
                doc = {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "dense_vector": chunk["embedding"],
                    # sparse_vector will be auto-generated by BM25 function
                    # Additional metadata fields
                    "url": chunk.get("metadata", {}).get("url", ""),
                    "title": chunk.get("metadata", {}).get("title", ""),
                    "chunk_index": chunk.get("metadata", {}).get("chunk_index", 0),
                    "source_hash": chunk.get("metadata", {}).get("source_hash", ""),
                }
                documents.append(doc)
            
            # Insert in batches
            batch_size = 100
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
                
                success = self.milvus_storage.insert_documents(
                    documents=batch,
                    partition_name=self.partition_name
                )
                
                if not success:
                    print(f"❌ Failed to insert batch {batch_num}")
                    return False
                
                time.sleep(0.1)  # Small delay between batches
            
            print(f"\n✅ Successfully stored {len(documents)} documents in Milvus")
            print(f"   - Dense vectors: {len(documents)}")
            print(f"   - Sparse vectors: Auto-generated by BM25")
            print(f"   - Partition: {self.partition_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error storing in Milvus: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete ingestion pipeline with Milvus storage."""
        try:
            print("\n" + "=" * 60)
            print("🚀 Starting Milvus Ingestion Pipeline")
            print("=" * 60)
            print(f"📍 Domain: {self.domain}")
            print(f"🏢 Partition: {self.partition_name}")
            print(f"📄 Max pages: {self.max_pages or 'unlimited'}")
            
            # Step 1: Discover URLs
            urls = self.discover_urls()
            if not urls:
                print("❌ No URLs discovered")
                return False
            
            # Step 2: Crawl pages
            pages = self.crawl_pages(urls)
            if not pages:
                print("❌ No content extracted from pages")
                return False
            
            # Step 3: Chunk content
            chunks = self.chunk_content(pages)
            if not chunks:
                print("❌ No chunks created from content")
                return False
            
            # Step 4: Generate embeddings
            chunks_with_embeddings = self.generate_embeddings(chunks)
            
            # Step 5: Store in Milvus (hybrid search with auto BM25)
            success = self.store_in_milvus(chunks_with_embeddings)
            
            if success:
                print("\n" + "=" * 60)
                print("🎉 Milvus Ingestion Pipeline Completed Successfully!")
                print(f"📊 Final stats:")
                print(f"   - URLs crawled: {len(pages)}")
                print(f"   - Chunks created: {len(chunks)}")
                print(f"   - Embeddings generated: {len(chunks_with_embeddings)}")
                print(f"   - Partition: {self.partition_name}")
                print(f"   - Dense vectors: {len(chunks_with_embeddings)}")
                print(f"   - Sparse vectors (BM25): Auto-generated")
                print(f"   - Hybrid search: ✅ Ready")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run Milvus ingestion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest website content into Milvus")
    parser.add_argument("--domain", help="Website domain to crawl")
    parser.add_argument("--partition", help="Partition name (e.g., org_zibtek)", default="_default")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to crawl")
    
    args = parser.parse_args()
    
    try:
        ingestor = MilvusIngestor(
            domain=args.domain,
            partition_name=args.partition,
            max_pages=args.max_pages
        )
        success = ingestor.run_full_pipeline()
        
        if success:
            print("\n✨ Ready for hybrid search (dense + BM25)!")
        else:
            print("\n❌ Ingestion failed. Check the errors above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Failed to initialize ingestor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# For testing Milvus ingestion
if __name__ == "__main__":
    main()
