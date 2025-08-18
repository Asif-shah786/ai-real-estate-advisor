"""
Aspect-Based Chunking Strategy Implementation

This file implements the Aspect-Based chunking strategy that was identified as the best performer
in our chunking strategy evaluation. It creates separate chunks for different aspects (crime, schools, transport)
and provides a vector database that can be integrated into the main RAG application.

Key Features:
- Separate chunks for crime, schools, transport aspects
- Optimized for specific aspect queries
- Better precision for focused questions
- Reduces noise in retrieval
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import openai
from dataclasses import dataclass
import time
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
from langchain_core.vectorstores import VectorStore

# Load environment variables
load_dotenv()

@dataclass
class AspectChunk:
    """Data class to store aspect-based chunks"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    aspect_type: str = "unknown"
    property_id: Optional[str] = None
    source_file: str = ""

class AspectBasedChunker:
    """
    Aspect-Based chunking system for Manchester Real Estate data
    
    This strategy creates separate chunks for different aspects (crime, schools, transport)
    which provides better precision for focused questions and reduces noise in retrieval.
    """
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.chunks = []
        self.vector_db: Optional[VectorStore] = None
        self.properties_data = []
        self.legal_data = []
        
    def load_data(self, properties_file: str, legal_file: str) -> Tuple[List[Dict], List[Dict]]:
        """Load the two JSON files containing property and legal data"""
        print(f"📁 Loading data files...")
        print(f"   Properties: {properties_file}")
        print(f"   Legal: {legal_file}")
        
        try:
            # Load properties data (JSON format)
            with open(properties_file, 'r', encoding='utf-8') as f:
                self.properties_data = json.load(f)
            print(f"✅ Loaded {len(self.properties_data)} properties")
            
            # Load legal data (JSONL format)
            self.legal_data = []
            with open(legal_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            legal_item = json.loads(line)
                            self.legal_data.append(legal_item)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Invalid JSON at line {line_num}: {e}")
                            continue
            print(f"✅ Loaded {len(self.legal_data)} legal entries")
            
            return self.properties_data, self.legal_data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return [], []
    
    def create_aspect_chunks(self) -> List[AspectChunk]:
        """
        Create aspect-based chunks for properties and legal data
        
        This method creates separate chunks for:
        - Crime information
        - School information  
        - Transport information
        - Legal information
        
        Returns:
            List of AspectChunk objects
        """
        print(f"\n🎯 Creating aspect-based chunks for {len(self.properties_data)} properties...")
        chunks = []
        
        for i, prop in enumerate(self.properties_data):
            if i % 10 == 0:  # Progress indicator
                print(f"   Processing property {i+1}/{len(self.properties_data)}...")
                
            base_info = f"Property at {prop.get('address', 'Unknown Address')}"
            if 'price' in prop:
                base_info += f", Price: £{prop['price']:,}"
            
            # Crime chunk
            if ('crime_data' in prop and prop['crime_data']) or ('crime_summary' in prop and prop['crime_summary']):
                crime_content = f"{base_info}\n\n"
                if 'crime_data' in prop and prop['crime_data']:
                    crime_content += f"Crime Information:\n{json.dumps(prop['crime_data'], indent=2)}"
                else:
                    crime_content += f"Crime Summary: {prop['crime_summary']}"
                
                chunks.append(AspectChunk(
                    chunk_id=f"property_{i}_crime",
                    content=crime_content,
                    metadata={
                        'type': 'crime',
                        'property_id': prop.get('property_id', i),
                        'address': prop.get('address', ''),
                        'postcode': prop.get('postcode', ''),
                        'price': prop.get('price', 0),
                        'property_type': prop.get('property_type', ''),
                        'bedrooms': prop.get('bedrooms', ''),
                        'source': prop.get('property_url', f"property_{i}"),
                        'source_title': prop.get('title', f"Property {i}")
                    },
                    aspect_type="crime",
                    property_id=prop.get('property_id', i),
                    source_file="properties"
                ))
            
            # Schools chunk
            if 'nearest_schools' in prop and prop['nearest_schools']:
                schools_content = f"{base_info}\n\nNearby Schools:\n{prop['nearest_schools']}"
                
                chunks.append(AspectChunk(
                    chunk_id=f"property_{i}_schools",
                    content=schools_content,
                    metadata={
                        'type': 'schools',
                        'property_id': prop.get('property_id', i),
                        'address': prop.get('address', ''),
                        'postcode': prop.get('postcode', ''),
                        'price': prop.get('price', 0),
                        'property_type': prop.get('property_type', ''),
                        'bedrooms': prop.get('bedrooms', ''),
                        'school_count': 1,
                        'source': prop.get('property_url', f"property_{i}"),
                        'source_title': prop.get('title', f"Property {i}")
                    },
                    aspect_type="schools",
                    property_id=prop.get('property_id', i),
                    source_file="properties"
                ))
            
            # Transport chunk
            if 'nearest_stations' in prop and prop['nearest_stations']:
                transport_content = f"{base_info}\n\nTransport Links:\n{prop['nearest_stations']}"
                
                chunks.append(AspectChunk(
                    chunk_id=f"property_{i}_transport",
                    content=transport_content,
                    metadata={
                        'type': 'transport',
                        'property_id': prop.get('property_id', i),
                        'address': prop.get('address', ''),
                        'postcode': prop.get('postcode', ''),
                        'price': prop.get('price', 0),
                        'property_type': prop.get('property_type', ''),
                        'bedrooms': prop.get('bedrooms', ''),
                        'transport_count': 1,
                        'source': prop.get('property_url', f"property_{i}"),
                        'source_title': prop.get('title', f"Property {i}")
                    },
                    aspect_type="transport",
                    property_id=prop.get('property_id', i),
                    source_file="properties"
                ))
            
            # Property overview chunk (basic info)
            overview_content = f"{base_info}\n\n"
            if 'description' in prop:
                overview_content += f"Description: {prop['description'][:300]}...\n\n"
            if 'bedrooms' in prop:
                overview_content += f"Bedrooms: {prop['bedrooms']}\n"
            if 'property_type' in prop:
                overview_content += f"Type: {prop['property_type']}\n"
            if 'tenure' in prop:
                overview_content += f"Tenure: {prop['tenure']}\n"
            if 'council_tax_band' in prop:
                overview_content += f"Council Tax: {prop['council_tax_band']}\n"
            
            chunks.append(AspectChunk(
                chunk_id=f"property_{i}_overview",
                content=overview_content,
                metadata={
                    'type': 'overview',
                    'property_id': prop.get('property_id', i),
                    'address': prop.get('address', ''),
                    'postcode': prop.get('postcode', ''),
                    'price': prop.get('price', 0),
                    'property_type': prop.get('property_type', ''),
                    'bedrooms': prop.get('bedrooms', ''),
                    'source': prop.get('property_url', f"property_{i}"),
                    'source_title': prop.get('title', f"Property {i}")
                },
                aspect_type="overview",
                property_id=prop.get('property_id', i),
                source_file="properties"
            ))
        
        # Add legal chunks
        print(f"\n⚖️  Creating legal chunks for {len(self.legal_data)} legal entries...")
        for i, legal_item in enumerate(self.legal_data):
            if i % 10 == 0:  # Progress indicator
                print(f"   Processing legal entry {i+1}/{len(self.legal_data)}...")
                
            # Create content from legal data
            content_parts = []
            
            if 'text' in legal_item:
                content_parts.append(f"Legal Information: {legal_item['text']}")
            
            if 'tags' in legal_item:
                content_parts.append(f"Tags: {', '.join(legal_item['tags'])}")
            
            if 'jurisdiction' in legal_item:
                content_parts.append(f"Jurisdiction: {legal_item['jurisdiction']}")
            
            content = "\n\n".join(content_parts)
            
            chunks.append(AspectChunk(
                chunk_id=f"legal_{i}",
                content=content,
                metadata={
                    'type': 'legal',
                    'category': legal_item.get('tags', ['general'])[0] if legal_item.get('tags') else 'general',
                    'topic': legal_item.get('id', ''),
                    'complexity': 'medium',
                    'jurisdiction': legal_item.get('jurisdiction', ''),
                    'source_name': legal_item.get('source_name', ''),
                    'source': legal_item.get('url', f"legal_{i}"),
                    'source_title': legal_item.get('id', f"Legal {i}")
                },
                aspect_type="legal",
                source_file="legal"
            ))
        
        self.chunks = chunks
        print(f"✅ Created {len(chunks)} aspect-based chunks")
        return chunks
    
    def generate_embeddings(self, model: str = "text-embedding-3-large") -> List[AspectChunk]:
        """
        Generate embeddings for chunks using OpenAI's embedding models
        
        Args:
            model: OpenAI embedding model to use
            
        Returns:
            List of AspectChunk objects with embeddings
        """
        print(f"\n🧠 Generating embeddings for {len(self.chunks)} chunks using {model}...")
        
        batch_size = 100  # OpenAI recommended batch size
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            texts = [chunk.content for chunk in batch]
            
            print(f"   Processing batch {i//batch_size + 1}/{(len(self.chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)...")
            
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=model
                )
                
                for j, chunk in enumerate(batch):
                    chunk.embedding = response.data[j].embedding
                    
                print(f"   ✅ Batch {i//batch_size + 1} completed successfully")
                
                # Rate limiting - be nice to OpenAI
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Error generating embeddings for batch {i//batch_size + 1}: {e}")
        
        print(f"✅ Embedding generation completed for {len(self.chunks)} chunks")
        return self.chunks
    
    def create_vector_database(self, embedding_model) -> Optional[VectorStore]:
        """
        Create a vector database from the aspect chunks
        
        Args:
            embedding_model: LangChain embedding model
            
        Returns:
            DocArrayInMemorySearch vector database
        """
        print(f"\n🗄️  Creating vector database with {len(self.chunks)} chunks...")
        
        # Convert AspectChunk to LangChain Document objects
        langchain_docs = []
        for chunk in self.chunks:
            if chunk.embedding:  # Only include chunks with embeddings
                doc = Document(
                    page_content=chunk.content,
                    metadata=chunk.metadata
                )
                langchain_docs.append(doc)
        
        print(f"   Chunks with embeddings: {len(langchain_docs)}")
        
        if not langchain_docs:
            print("   ❌ No chunks with embeddings found!")
            return None
        
        try:
            # Create vector database using DocArrayInMemorySearch
            vector_db = DocArrayInMemorySearch.from_documents(langchain_docs, embedding_model)
            print(f"✅ Vector database created successfully!")
            self.vector_db = vector_db
            return vector_db
            
        except Exception as e:
            print(f"   ❌ Error creating DocArrayInMemorySearch: {e}")
            print(f"   🔄 Falling back to Chroma...")
            
            try:
                # Fallback to Chroma
                vector_db = Chroma.from_documents(langchain_docs, embedding_model)
                print(f"✅ Chroma vector database created successfully!")
                self.vector_db = vector_db
                return vector_db
                
            except Exception as e2:
                print(f"   ❌ Error creating Chroma database: {e2}")
                return None
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[AspectChunk, float]]:
        """
        Search for similar chunks in the vector database
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (chunk, similarity_score)
        """
        if not self.vector_db:
            print("❌ Vector database not available!")
            return []
        
        print(f"\n🔍 Searching for: {query}")
        
        try:
            # Search in vector database using LangChain's similarity search
            results = self.vector_db.similarity_search_with_score(query, k=top_k)
            
            # Convert to AspectChunk format
            chunk_results = []
            for doc, score in results:
                # Find corresponding AspectChunk
                for chunk in self.chunks:
                    if chunk.content == doc.page_content:
                        chunk_results.append((chunk, score))
                        break
            
            print(f"✅ Found {len(chunk_results)} results")
            return chunk_results
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get statistics about the created chunks"""
        if not self.chunks:
            return {}
        
        aspect_counts = {}
        for chunk in self.chunks:
            aspect_type = chunk.aspect_type
            aspect_counts[aspect_type] = aspect_counts.get(aspect_type, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'aspect_distribution': aspect_counts,
            'chunks_with_embeddings': len([c for c in self.chunks if c.embedding]),
            'avg_chunk_size': np.mean([len(chunk.content.split()) for chunk in self.chunks])
        }
    
    def save_chunks_to_jsonl(self, output_path: str):
        """Save chunks to JSONL format for persistence"""
        print(f"\n💾 Saving chunks to {output_path}...")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in self.chunks:
                    # Prepare chunk data for JSON serialization
                    chunk_data = {
                        'chunk_id': chunk.chunk_id,
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'aspect_type': chunk.aspect_type,
                        'property_id': chunk.property_id,
                        'source_file': chunk.source_file
                    }
                    f.write(json.dumps(chunk_data) + '\n')
            
            print(f"✅ Saved {len(self.chunks)} chunks to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving chunks: {e}")

def create_aspect_based_vectordb(openai_api_key: str, 
                                properties_file: str = "dataset_v2/properties_with_crime_data.json",
                                legal_file: str = "dataset_v2/legal_uk_greater_manchester.jsonl",
                                embedding_model=None) -> Optional[VectorStore]:
    """
    Factory function to create an aspect-based vector database
    
    Args:
        openai_api_key: OpenAI API key
        properties_file: Path to properties JSON file
        legal_file: Path to legal JSONL file
        embedding_model: LangChain embedding model (optional)
        
    Returns:
        DocArrayInMemorySearch vector database
    """
    print("🚀 Creating Aspect-Based Vector Database...")
    
    # Initialize chunker
    chunker = AspectBasedChunker(openai_api_key)
    
    # Load data
    properties_data, legal_data = chunker.load_data(properties_file, legal_file)
    if not properties_data and not legal_data:
        print("❌ Failed to load data!")
        return None
    
    # Create chunks
    chunks = chunker.create_aspect_chunks()
    
    # Generate embeddings
    chunks_with_embeddings = chunker.generate_embeddings()
    
    # Create vector database
    if embedding_model is None:
        print("⚠️  No embedding model provided, using OpenAI embeddings directly")
        # Create a simple embedding function that uses OpenAI
        def openai_embedding_function(texts):
            if isinstance(texts, str):
                texts = [texts]
            embeddings = []
            for text in texts:
                response = chunker.client.embeddings.create(
                    input=[text],
                    model="text-embedding-3-large"
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        
        # Create a mock embedding model object
        class MockEmbeddingModel:
            def embed_documents(self, texts):
                return openai_embedding_function(texts)
            
            def embed_query(self, text):
                return openai_embedding_function(text)[0]
        
        embedding_model = MockEmbeddingModel()
    
    vector_db = chunker.create_vector_database(embedding_model)
    
    # Show statistics
    stats = chunker.get_chunk_statistics()
    print(f"\n📊 Chunk Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Aspect distribution: {stats['aspect_distribution']}")
    print(f"   Chunks with embeddings: {stats['chunks_with_embeddings']}")
    print(f"   Average chunk size: {stats['avg_chunk_size']:.1f} words")
    
    return vector_db

# Example usage
if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables!")
        print("   Please set your OpenAI API key in the .env file")
        exit(1)
    
    print("🧠 Testing Aspect-Based Chunking System...")
    
    # Create vector database
    vector_db = create_aspect_based_vectordb(api_key)
    
    if vector_db:
        # Test search functionality
        chunker = AspectBasedChunker(api_key)
        chunker.vector_db = vector_db
        
        # Test queries
        test_queries = [
            "Properties with low crime rates",
            "Good schools in Manchester",
            "Transport links near properties",
            "Legal requirements for buying property"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: {query}")
            results = chunker.search_similar(query, top_k=3)
            
            for i, (chunk, score) in enumerate(results, 1):
                print(f"   Result {i} (Score: {score:.4f}):")
                print(f"      Type: {chunk.aspect_type}")
                print(f"      Content: {chunk.content[:100]}...")
                print()
        
        print("✅ Aspect-based chunking system test completed!")
    else:
        print("❌ Failed to create vector database!")
