"""Milvus vector storage helpers for the Zibtek chatbot.

This module provides Milvus/Zilliz Cloud integration with hybrid search support.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from pymilvus import MilvusClient, DataType, Function, FunctionType

load_dotenv()


# ============================================================================
# MILVUS STORAGE
# ============================================================================

class MilvusStorage:
    """Milvus vector storage client with hybrid search support."""
    
    def __init__(self):
        """Initialize Milvus client."""
        self.uri = os.getenv("MILVUS_URI")
        self.token = os.getenv("MILVUS_TOKEN")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "documents")
        
        if not self.uri or not self.token:
            print("⚠️  MILVUS_URI or MILVUS_TOKEN not found. Milvus operations will be disabled.")
            self.client = None
            return
        
        try:
            self.client = MilvusClient(uri=self.uri, token=self.token)
            print(f"✅ Milvus client initialized")
            print(f"   - URI: {self.uri}")
            print(f"   - Collection: {self.collection_name}")
        except Exception as e:
            print(f"❌ Failed to initialize Milvus client: {e}")
            self.client = None
    
    def init_collection(self, dimension: int = 1536) -> bool:
        """Initialize or connect to Milvus collection with hybrid search support."""
        if not self.client:
            print("❌ Milvus client not initialized")
            return False
        
        try:
            # Check if collection exists
            if self.collection_name in self.client.list_collections():
                print(f"✅ Collection '{self.collection_name}' already exists")
                return True
            
            print(f"🏗️  Creating collection '{self.collection_name}' with hybrid search support...")
            
            # Define schema
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True  # Allow flexible metadata fields
            )
            
            # Add fields
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(
                field_name="text",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True  # Required for BM25 function
            )
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            
            # Add BM25 function (auto-generates sparse_vector from text)
            bm25_function = Function(
                name="text_bm25_emb",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
                params={},
            )
            schema.add_function(bm25_function)
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
            )
            
            # Create indices
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25"
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            print(f"✅ Collection '{self.collection_name}' created with hybrid search support")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing collection: {e}")
            return False
    
    def create_partition(self, partition_name: str) -> bool:
        """Create a partition for multi-tenant support.
        
        Args:
            partition_name: Name of the partition (e.g., "org_zibtek")
        
        Returns:
            True if partition created or already exists
        """
        if not self.client:
            print("❌ Milvus client not initialized")
            return False
        
        try:
            # Check if partition already exists
            partitions = self.client.list_partitions(self.collection_name)
            if partition_name in partitions:
                print(f"✅ Partition '{partition_name}' already exists")
                return True
            
            # Create partition
            self.client.create_partition(
                collection_name=self.collection_name,
                partition_name=partition_name
            )
            print(f"✅ Created partition: {partition_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating partition: {e}")
            return False
    
    def insert_documents(
        self,
        documents: List[Dict[str, Any]],
        partition_name: str = "_default"
    ) -> bool:
        """Insert documents into Milvus collection.
        
        Args:
            documents: List of documents with fields: id, text, dense_vector
                      (sparse_vector will be auto-generated from text)
            partition_name: Partition name for multi-tenant isolation
        
        Returns:
            True if successful
        """
        if not self.client:
            print("❌ Milvus client not initialized")
            return False
        
        try:
            # Ensure partition exists
            self.create_partition(partition_name)
            
            # Insert documents
            self.client.insert(
                collection_name=self.collection_name,
                data=documents,
                partition_name=partition_name
            )
            
            print(f"✅ Inserted {len(documents)} documents into partition '{partition_name}'")
            return True
            
        except Exception as e:
            print(f"❌ Error inserting documents: {e}")
            return False
    
    def get_collection_stats(self, partition_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collection or partition statistics.
        
        Args:
            partition_name: If provided, get stats for specific partition
        
        Returns:
            Dictionary with statistics
        """
        if not self.client:
            return {"error": "Milvus client not initialized"}
        
        try:
            # Get collection info
            stats = self.client.describe_collection(self.collection_name)
            
            # List all partitions
            partitions = self.client.list_partitions(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "total_entities": stats.get("num_entities", 0),
                "partitions": partitions,
                "dimension": 1536,
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def delete_partition(self, partition_name: str, confirm: bool = False) -> bool:
        """Delete a partition (use with caution).
        
        Args:
            partition_name: Name of partition to delete
            confirm: Must be True to proceed
        
        Returns:
            True if successful
        """
        if not confirm:
            print("❌ Must set confirm=True to delete partition")
            return False
        
        if not self.client:
            print("❌ Milvus client not initialized")
            return False
        
        try:
            self.client.drop_partition(
                collection_name=self.collection_name,
                partition_name=partition_name
            )
            print(f"✅ Deleted partition '{partition_name}'")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting partition: {e}")
            return False


# Global Milvus instance
milvus_storage = MilvusStorage()


# Helper functions for Milvus
def init_milvus_collection(dimension: int = 1536) -> bool:
    """Initialize Milvus collection."""
    return milvus_storage.init_collection(dimension)


def create_milvus_partition(partition_name: str) -> bool:
    """Create a Milvus partition for multi-tenant support."""
    return milvus_storage.create_partition(partition_name)


def insert_milvus_documents(
    documents: List[Dict[str, Any]],
    partition_name: str = "_default"
) -> bool:
    """Insert documents into Milvus."""
    return milvus_storage.insert_documents(documents, partition_name)


def get_milvus_stats(partition_name: Optional[str] = None) -> Dict[str, Any]:
    """Get Milvus collection statistics."""
    return milvus_storage.get_collection_stats(partition_name)
