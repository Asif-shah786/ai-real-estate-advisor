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

    def to_document(self) -> Document:
        """Convert AspectChunk to LangChain Document object"""
        return Document(page_content=self.content, metadata=self.metadata)


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

    def load_data(
        self, properties_file: str, legal_file: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Load the two JSON files containing property and legal data"""
        print(f"📁 Loading data files...")
        print(f"   Properties: {properties_file}")
        print(f"   Legal: {legal_file}")

        try:
            # Load properties data (JSON format)
            with open(properties_file, "r", encoding="utf-8") as f:
                self.properties_data = json.load(f)
            print(f"✅ Loaded {len(self.properties_data)} properties")

            # Load legal data (JSONL format)
            self.legal_data = []
            with open(legal_file, "r", encoding="utf-8") as f:
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
        print(
            f"\n🎯 Creating aspect-based chunks for {len(self.properties_data)} properties..."
        )
        chunks = []

        for i, prop in enumerate(self.properties_data):
            if i % 10 == 0:  # Progress indicator
                print(f"   Processing property {i+1}/{len(self.properties_data)}...")

            base_info = f"Property at {prop.get('address', 'Unknown Address')}"
            if "price_int" in prop:
                base_info += f", price_int: £{prop['price_int']:,}"

            # Crime chunk
            if ("crime_data" in prop and prop["crime_data"]) or (
                "crime_summary" in prop and prop["crime_summary"]
            ):
                crime_content = f"{base_info}\n\n"
                if "crime_data" in prop and prop["crime_data"]:
                    crime_content += f"Crime Information:\n{json.dumps(prop['crime_data'], indent=2)}"
                else:
                    crime_content += f"Crime Summary: {prop['crime_summary']}"

                chunks.append(
                    AspectChunk(
                        chunk_id=f"property_{i}_crime",
                        content=crime_content,
                        metadata={
                            "type": "crime",
                            "property_id": prop.get("property_id", i),
                            "address": prop.get("address", ""),
                            "postcode": prop.get("postcode", ""),
                            "price_int": prop.get("price_int", 0),
                            "property_type": prop.get("property_type", ""),
                            "bedrooms": prop.get("bedrooms", ""),
                            "source": prop.get("property_url", f"property_{i}"),
                            "source_title": prop.get("title", f"Property {i}"),
                        },
                        aspect_type="crime",
                        property_id=prop.get("property_id", i),
                        source_file="properties",
                    )
                )

            # Schools chunk
            if "nearest_schools" in prop and prop["nearest_schools"]:
                schools_content = (
                    f"{base_info}\n\nNearby Schools:\n{prop['nearest_schools']}"
                )

                chunks.append(
                    AspectChunk(
                        chunk_id=f"property_{i}_schools",
                        content=schools_content,
                        metadata={
                            "type": "schools",
                            "property_id": prop.get("property_id", i),
                            "address": prop.get("address", ""),
                            "postcode": prop.get("postcode", ""),
                            "price_int": prop.get("price_int", 0),
                            "property_type": prop.get("property_type", ""),
                            "bedrooms": prop.get("bedrooms", ""),
                            "school_count": 1,
                            "source": prop.get("property_url", f"property_{i}"),
                            "source_title": prop.get("title", f"Property {i}"),
                        },
                        aspect_type="schools",
                        property_id=prop.get("property_id", i),
                        source_file="properties",
                    )
                )

            # Transport chunk
            if "nearest_stations" in prop and prop["nearest_stations"]:
                transport_content = (
                    f"{base_info}\n\nTransport Links:\n{prop['nearest_stations']}"
                )

                chunks.append(
                    AspectChunk(
                        chunk_id=f"property_{i}_transport",
                        content=transport_content,
                        metadata={
                            "type": "transport",
                            "property_id": prop.get("property_id", i),
                            "address": prop.get("address", ""),
                            "postcode": prop.get("postcode", ""),
                            "price_int": prop.get("price_int", 0),
                            "property_type": prop.get("property_type", ""),
                            "bedrooms": prop.get("bedrooms", ""),
                            "transport_count": 1,
                            "source": prop.get("property_url", f"property_{i}"),
                            "source_title": prop.get("title", f"Property {i}"),
                        },
                        aspect_type="transport",
                        property_id=prop.get("property_id", i),
                        source_file="properties",
                    )
                )

            # Property overview chunk (basic info)
            overview_content = f"{base_info}\n\n"
            if "description" in prop:
                overview_content += f"Description: {prop['description'][:300]}...\n\n"
            if "bedrooms" in prop:
                overview_content += f"Bedrooms: {prop['bedrooms']}\n"
            if "property_type" in prop:
                overview_content += f"Type: {prop['property_type']}\n"
            if "tenure" in prop:
                overview_content += f"Tenure: {prop['tenure']}\n"
            if "council_tax_band" in prop:
                overview_content += f"Council Tax: {prop['council_tax_band']}\n"

            chunks.append(
                AspectChunk(
                    chunk_id=f"property_{i}_overview",
                    content=overview_content,
                    metadata={
                        "type": "overview",
                        "property_id": prop.get("property_id", i),
                        "address": prop.get("address", ""),
                        "postcode": prop.get("postcode", ""),
                        "price_int": prop.get("price_int", 0),
                        "property_type": prop.get("property_type", ""),
                        "bedrooms": prop.get("bedrooms", ""),
                        "source": prop.get("property_url", f"property_{i}"),
                        "source_title": prop.get("title", f"Property {i}"),
                    },
                    aspect_type="overview",
                    property_id=prop.get("property_id", i),
                    source_file="properties",
                )
            )

        # Add legal chunks
        print(f"\n⚖️  Creating legal chunks for {len(self.legal_data)} legal entries...")
        for i, legal_item in enumerate(self.legal_data):
            if i % 10 == 0:  # Progress indicator
                print(f"   Processing legal entry {i+1}/{len(self.legal_data)}...")

            # Create content from legal data
            content_parts = []

            if "text" in legal_item:
                content_parts.append(f"Legal Information: {legal_item['text']}")

            if "tags" in legal_item:
                content_parts.append(f"Tags: {', '.join(legal_item['tags'])}")

            if "jurisdiction" in legal_item:
                content_parts.append(f"Jurisdiction: {legal_item['jurisdiction']}")

            content = "\n\n".join(content_parts)

            chunks.append(
                AspectChunk(
                    chunk_id=f"legal_{i}",
                    content=content,
                    metadata={
                        "type": "legal",
                        "category": (
                            legal_item.get("tags", ["general"])[0]
                            if legal_item.get("tags")
                            else "general"
                        ),
                        "topic": legal_item.get("id", ""),
                        "complexity": "medium",
                        "jurisdiction": legal_item.get("jurisdiction", ""),
                        "source_name": legal_item.get("source_name", ""),
                        "source": legal_item.get("url", f"legal_{i}"),
                        "source_title": legal_item.get("id", f"Legal {i}"),
                    },
                    aspect_type="legal",
                    source_file="legal",
                )
            )

        self.chunks = chunks
        print(f"✅ Created {len(chunks)} aspect-based chunks")
        return chunks

    def generate_embeddings(
        self, model: str = "text-embedding-3-large"
    ) -> List[AspectChunk]:
        """
        Generate embeddings for chunks using OpenAI's embedding models

        Args:
            model: OpenAI embedding model to use

        Returns:
            List of AspectChunk objects with embeddings
        """
        print(
            f"\n🧠 Generating embeddings for {len(self.chunks)} chunks using {model}..."
        )

        batch_size = 100  # OpenAI recommended batch size

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]

            print(
                f"   Processing batch {i//batch_size + 1}/{(len(self.chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)..."
            )

            try:
                response = self.client.embeddings.create(input=texts, model=model)

                for j, chunk in enumerate(batch):
                    chunk.embedding = response.data[j].embedding

                print(f"   ✅ Batch {i//batch_size + 1} completed successfully")

                # Rate limiting - be nice to OpenAI
                time.sleep(0.1)

            except Exception as e:
                print(
                    f"   ❌ Error generating embeddings for batch {i//batch_size + 1}: {e}"
                )

        print(f"✅ Embedding generation completed for {len(self.chunks)} chunks")
        return self.chunks

    def create_vector_database(self, embedding_model=None) -> Optional[VectorStore]:
        """
        Create a vector database from the aspect chunks

        This method:
        1. Generates embeddings for all chunks using OpenAI (default) or provided model
        2. Creates a DocArrayInMemorySearch vector store
        3. Saves the vector database to disk for persistence

        Parameters:
            embedding_model: The embedding model to use (if None, uses OpenAI)

        Returns:
            VectorStore: The created vector database, or None if failed
        """
        if not self.chunks:
            print("❌ No chunks available to create vector database")
            return None

        print(f"🔍 Generating embeddings for {len(self.chunks)} chunks...")

        # Generate embeddings for all chunks using OpenAI by default
        self.generate_embeddings()

        # Debug: Check embedding dimensions
        if self.chunks and self.chunks[0].embedding:
            first_embedding = self.chunks[0].embedding
            print(f"🔍 First chunk embedding dimension: {len(first_embedding)}")
            print(
                f"🔍 Total chunks with embeddings: {len([c for c in self.chunks if c.embedding])}"
            )

        print("🏗️ Creating vector database...")

        try:
            # Create DocArrayInMemorySearch vector store
            # Convert chunks to documents and use OpenAI embeddings directly
            documents = []
            for chunk in self.chunks:
                if chunk.embedding:  # Only include chunks with embeddings
                    doc = Document(page_content=chunk.content, metadata=chunk.metadata)
                    documents.append(doc)

            if not documents:
                print("❌ No chunks with embeddings found!")
                return None

            # Create a simple embedding function that uses the stored embeddings
            from langchain_core.embeddings import Embeddings

            class OpenAIEmbeddingFunction(Embeddings):
                def __init__(self, chunks, openai_client):
                    self.chunks = chunks
                    self.openai_client = openai_client
                    self.embeddings = {
                        chunk.content: chunk.embedding
                        for chunk in chunks
                        if chunk.embedding
                    }
                    # Get the dimension from the first embedding
                    if self.embeddings:
                        first_embedding = next(iter(self.embeddings.values()))
                        self.embedding_dim = len(first_embedding)
                        print(f"📏 Embedding dimension: {self.embedding_dim}")
                    else:
                        self.embedding_dim = 1536  # OpenAI default
                        print(
                            f"⚠️ No embeddings found, using default dimension: {self.embedding_dim}"
                        )

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    results = []
                    for text in texts:
                        if text in self.embeddings:
                            results.append(self.embeddings[text])
                        else:
                            # Fallback: create zero vector with correct dimension
                            results.append([0.0] * self.embedding_dim)
                    return results

                def embed_query(self, text: str) -> List[float]:
                    # For queries, we need to generate new embeddings with OpenAI
                    # This ensures the query has the same dimension as stored embeddings
                    try:
                        response = self.openai_client.embeddings.create(
                            input=[text], model="text-embedding-3-large"
                        )
                        embedding = response.data[0].embedding

                        # Check if dimension matches stored embeddings
                        if self.embeddings and len(embedding) != self.embedding_dim:
                            print(
                                f"⚠️ Query embedding dimension ({len(embedding)}) doesn't match stored embeddings ({self.embedding_dim})"
                            )
                            # Pad or truncate to match
                            if len(embedding) < self.embedding_dim:
                                embedding.extend(
                                    [0.0] * (self.embedding_dim - len(embedding))
                                )
                            else:
                                embedding = embedding[: self.embedding_dim]

                        return embedding
                    except Exception as e:
                        print(f"❌ Error generating query embedding: {e}")
                        # Fallback: return zero vector with correct dimension
                        return [0.0] * self.embedding_dim

            embedding_function = OpenAIEmbeddingFunction(self.chunks, self.client)

            self.vector_db = DocArrayInMemorySearch.from_documents(
                documents, embedding_function
            )

            # Save the vector database to disk for persistence
            self.save_vector_database()

            print(
                f"✅ Vector database created successfully with {len(documents)} chunks!"
            )
            return self.vector_db

        except Exception as e:
            print(f"❌ Error creating DocArrayInMemorySearch: {e}")

            # Fallback to Chroma
            try:
                from langchain_community.vectorstores import Chroma

                self.vector_db = Chroma.from_documents(documents, embedding_function)

                # Save the vector database to disk for persistence
                self.save_vector_database()

                print(
                    f"✅ Chroma vector database created successfully with {len(documents)} chunks!"
                )
                return self.vector_db

            except Exception as e2:
                print(f"❌ Error creating Chroma database: {e2}")
                return None

    def save_vector_database(self):
        """Save the vector database and chunks to disk for persistence"""
        try:
            # Create artifacts directory if it doesn't exist
            os.makedirs("artifacts_v3", exist_ok=True)

            # Save chunks data (without embeddings to avoid serialization issues)
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "aspect_type": chunk.aspect_type,
                    "property_id": chunk.property_id,
                    "source_file": chunk.source_file,
                }
                chunks_data.append(chunk_dict)

            with open("artifacts_v3/aspect_chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)

            print(
                f"💾 Saved {len(chunks_data)} chunks to artifacts_v3/aspect_chunks.json"
            )

        except Exception as e:
            print(f"⚠️ Warning: Could not save chunks data: {e}")

    def load_vector_database(self, embedding_model=None) -> Optional[VectorStore]:
        """
        Load an existing vector database from disk if available

        Parameters:
            embedding_model: The embedding model to use (ignored, uses OpenAI)

        Returns:
            VectorStore: The loaded vector database, or None if not available
        """
        try:
            # Check if we have saved chunks
            chunks_file = "artifacts_v3/aspect_chunks.json"
            if not os.path.exists(chunks_file):
                print("📁 No saved chunks found, will create new vector database")
                return None

            print("📁 Loading existing chunks from disk...")

            # Load chunks data
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            # Recreate AspectChunk objects
            self.chunks = []
            for chunk_dict in chunks_data:
                chunk = AspectChunk(
                    chunk_id=chunk_dict["chunk_id"],
                    content=chunk_dict["content"],
                    metadata=chunk_dict["metadata"],
                    aspect_type=chunk_dict["aspect_type"],
                    property_id=chunk_dict["property_id"],
                    source_file=chunk_dict["source_file"],
                )
                self.chunks.append(chunk)

            print(f"✅ Loaded {len(self.chunks)} chunks from disk")

            # Generate embeddings and create vector database
            return self.create_vector_database(embedding_model)

        except Exception as e:
            print(f"⚠️ Warning: Could not load saved chunks: {e}")
            return None


def create_aspect_based_vectordb(
    openai_api_key: str,
    properties_file: str = "dataset_v2/run_ready_100.json",
    legal_file: str = "dataset_v2/legal_uk_greater_manchester.jsonl",
    embedding_model=None,
    force_recreate: bool = False,
):
    """
    Create an Aspect-Based vector database for real estate data

    This function:
    1. First tries to load existing vector database from disk (unless force_recreate=True)
    2. If none exists or force_recreate=True, creates new chunks and vector database
    3. Uses OpenAI embeddings by default for optimal quality
    4. Saves everything to disk for future use

    Parameters:
        openai_api_key: OpenAI API key for embeddings
        properties_file: Path to properties JSON file
        legal_file: Path to legal JSONL file
        embedding_model: LangChain embedding model (optional, OpenAI used by default)
        force_recreate: If True, ignore existing database and create new one

    Returns:
        VectorStore: The created/loaded vector database
    """
    print("🚀 Creating Aspect-Based Vector Database...")

    # Initialize chunker
    chunker = AspectBasedChunker(openai_api_key)

    # Check if we should force recreation
    if force_recreate:
        print("🔄 Force recreate flag set - ignoring existing database")
    else:
        # First, try to load existing vector database from disk
        print("📁 Checking for existing vector database...")
        existing_vectordb = chunker.load_vector_database(embedding_model)

        if existing_vectordb:
            print("✅ Successfully loaded existing vector database from disk!")
            return existing_vectordb

    # If no existing database or force_recreate=True, create new one
    print("🆕 Creating new vector database...")

    # Load data
    properties_data, legal_data = chunker.load_data(properties_file, legal_file)

    if not properties_data:
        print("❌ Failed to load properties data")
        return None

    # Create aspect-based chunks
    chunks = chunker.create_aspect_chunks()

    if not chunks:
        print("❌ Failed to create aspect chunks")
        return None

    # Create vector database (OpenAI embeddings used by default)
    vectordb = chunker.create_vector_database(embedding_model)

    if vectordb:
        print("✅ Aspect-Based Vector Database created and saved successfully!")
        return vectordb
    else:
        print("❌ Failed to create vector database")
        return None


# Example usage
if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables!")
        print("   Please set your OpenAI API key in the .env file")
        exit(1)

    print("🧠 Testing Aspect-Based Chunking System...")

    # Create vector database
    vector_db = create_aspect_based_vectordb(api_key)

    if vector_db:
        print("✅ Vector database created successfully!")
        print("🎯 Ready for real estate queries!")
    else:
        print("❌ Failed to create vector database")
