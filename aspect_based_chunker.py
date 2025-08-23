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
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore

# Define constants directly instead of importing from app.py to avoid circular dependency
LOCAL_DATASET_PATH_LEGAL_JSONL = "../datasets/legal_uk_greater_manchester.jsonl"
LOCAL_DATASET_PATH_LISTING_JSON = "../datasets/run_ready_904.json"

import utils

# Load environment variables
load_dotenv()


def filter_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter metadata to remove None values and convert to Chroma-compatible types.

    CRITICAL: Ensure metadata fields are properly typed for SelfQueryRetriever filtering.

    Args:
        metadata: Original metadata dictionary

    Returns:
        Filtered metadata dictionary with only str, int, float, bool values
    """
    filtered = {}
    for key, value in metadata.items():
        if value is not None:
            # CRITICAL: Handle specific fields for proper filtering
            if key == "postcode":
                # Ensure postcode is a clean string for filtering
                filtered[key] = str(value).strip().upper()
            elif key == "price_int":
                # Ensure price is an integer for range filtering
                try:
                    filtered[key] = int(float(value)) if value else 0
                except (ValueError, TypeError):
                    filtered[key] = 0
            elif key == "bedrooms":
                # Ensure bedrooms is a number for filtering
                try:
                    filtered[key] = float(value) if value else 0.0
                except (ValueError, TypeError):
                    filtered[key] = 0.0
            elif key == "property_type":
                # Ensure property type is a clean string for filtering
                filtered[key] = str(value).strip().lower() if value else ""
            elif key == "tenure":
                # Ensure tenure is a clean string for filtering
                filtered[key] = str(value).strip().lower() if value else ""
            elif isinstance(value, (str, int, float, bool)):
                # Handle other basic types
                filtered[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to strings but preserve key info
                if key == "tags" and isinstance(value, list):
                    # For tags, join into a searchable string
                    filtered[key] = ", ".join(str(tag) for tag in value)
                else:
                    filtered[key] = str(value)
            else:
                # Convert other types to strings
                filtered[key] = str(value)

    # CRITICAL: Add validation that required fields exist
    required_fields = ["postcode", "price_int", "property_type", "bedrooms"]
    for field in required_fields:
        if field not in filtered:
            print(f"⚠️ Warning: Missing required metadata field: {field}")
            if field == "postcode":
                filtered[field] = "UNKNOWN"
            elif field == "price_int":
                filtered[field] = 0
            elif field == "property_type":
                filtered[field] = "unknown"
            elif field == "bedrooms":
                filtered[field] = 0.0

    return filtered


def get_database_name(properties_file: str, legal_file: str) -> str:
    """
    Generate a unique database name based on dataset files

    Args:
        properties_file: Path to properties JSON file
        legal_file: Path to legal JSONL file

    Returns:
        str: Unique database name based on file names
    """
    # Extract base names without extensions
    prop_name = Path(properties_file).stem
    legal_name = Path(legal_file).stem

    # Create a unique identifier
    db_name = f"{prop_name}_{legal_name}"

    # Clean the name to make it filesystem-safe
    db_name = "".join(c for c in db_name if c.isalnum() or c in ["_", "-"])

    return db_name


def check_files_modified(properties_file: str, legal_file: str, db_path: str) -> bool:
    """
    Check if dataset files have been modified since database was created

    Args:
        properties_file: Path to properties JSON file
        legal_file: Path to legal JSONL file
        db_path: Path to database directory

    Returns:
        bool: True if files have been modified, False otherwise
    """
    try:
        if not os.path.exists(db_path):
            return True  # Database doesn't exist, consider it modified

        db_modified_time = os.path.getmtime(db_path)

        # Check if either file is newer than the database
        for file_path in [properties_file, legal_file]:
            if os.path.exists(file_path):
                file_modified_time = os.path.getmtime(file_path)
                if file_modified_time > db_modified_time:
                    return True

        return False
    except Exception:
        return True  # If we can't check, assume files are modified


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
                        metadata=filter_metadata(
                            {
                                "type": "crime",
                                "property_id": prop.get("property_id", i),
                                "address": prop.get("address", ""),
                                "postcode": prop.get("postcode", ""),
                                "price_int": prop.get("price_int", 0),
                                "property_type": prop.get("property_type", ""),
                                "bedrooms": prop.get("bedrooms", ""),
                                "source": prop.get("property_url", f"property_{i}"),
                                "source_title": prop.get("title", f"Property {i}"),
                            }
                        ),
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
                        metadata=filter_metadata(
                            {
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
                            }
                        ),
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
                        metadata=filter_metadata(
                            {
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
                            }
                        ),
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
                    metadata=filter_metadata(
                        {
                            "type": "overview",
                            "property_id": prop.get("property_id", i),
                            "address": prop.get("address", ""),
                            "postcode": prop.get("postcode", ""),
                            "price_int": prop.get("price_int", 0),
                            "property_type": prop.get("property_type", ""),
                            "bedrooms": prop.get("bedrooms", ""),
                            "source": prop.get("property_url", f"property_{i}"),
                            "source_title": prop.get("title", f"Property {i}"),
                        }
                    ),
                    aspect_type="overview",
                    property_id=prop.get("property_id", i),
                    source_file="properties",
                )
            )

        # Validate metadata consistency across chunks
        print(f"\n🔍 Validating metadata consistency...")
        self._validate_chunk_metadata(chunks)

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
                    metadata=filter_metadata(
                        {
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
                        }
                    ),
                    aspect_type="legal",
                    source_file="legal",
                )
            )

        self.chunks = chunks
        print(f"✅ Created {len(chunks)} aspect-based chunks")
        return chunks

    def _validate_chunk_metadata(self, chunks: List[AspectChunk]):
        """
        Validate that all chunks have consistent and properly typed metadata.

        This is critical for SelfQueryRetriever to work properly.
        """
        print(f"   Validating {len(chunks)} chunks...")

        # Check metadata field consistency
        metadata_fields = set()
        postcodes = set()
        price_ranges = []
        property_types = set()

        for chunk in chunks:
            if hasattr(chunk, "metadata") and chunk.metadata:
                metadata_fields.update(chunk.metadata.keys())

                # Check postcode format
                if "postcode" in chunk.metadata:
                    postcode = chunk.metadata["postcode"]
                    postcodes.add(postcode)
                    if not isinstance(postcode, str) or len(postcode) < 2:
                        print(
                            f"   ⚠️ Invalid postcode in chunk {chunk.chunk_id}: {postcode}"
                        )

                # Check price format
                if "price_int" in chunk.metadata:
                    price = chunk.metadata["price_int"]
                    if isinstance(price, (int, float)):
                        price_ranges.append(price)
                    else:
                        print(f"   ⚠️ Invalid price in chunk {chunk.chunk_id}: {price}")

                # Check property type format
                if "property_type" in chunk.metadata:
                    prop_type = chunk.metadata["property_type"]
                    if isinstance(prop_type, str):
                        property_types.add(prop_type)
                    else:
                        print(
                            f"   ⚠️ Invalid property_type in chunk {chunk.chunk_id}: {prop_type}"
                        )

        print(f"   ✅ Metadata fields found: {sorted(metadata_fields)}")
        print(
            f"   ✅ Postcodes found: {sorted(postcodes)[:10]}{'...' if len(postcodes) > 10 else ''}"
        )
        print(
            f"   ✅ Price range: £{min(price_ranges):,} - £{max(price_ranges):,}"
            if price_ranges
            else "   ⚠️ No valid prices found"
        )
        print(f"   ✅ Property types: {sorted(property_types)}")

        # Validate required fields exist
        required_fields = ["postcode", "price_int", "property_type", "bedrooms"]
        missing_fields = [
            field for field in required_fields if field not in metadata_fields
        ]
        if missing_fields:
            print(f"   ❌ Missing required metadata fields: {missing_fields}")
        else:
            print(f"   ✅ All required metadata fields present")

    def create_vector_database(
        self, properties_file: str, legal_file: str, embedding_model=None
    ) -> Optional[VectorStore]:
        """
        Create a vector database from the aspect chunks with file-based naming

        This method:
        1. Generates embeddings for all chunks using OpenAI (default) or provided model
        2. Creates a Chroma vector store with persistent storage
        3. Uses file-based naming for the database
        4. Saves the vector database to disk for persistence

        Parameters:
            properties_file: Path to properties JSON file (for database naming)
            legal_file: Path to legal JSONL file (for database naming)
            embedding_model: The embedding model to use (if None, uses OpenAI)

        Returns:
            VectorStore: The created vector database, or None if failed
        """
        if not self.chunks:
            print("❌ No chunks available to create vector database")
            return None

        print(f"🔍 Generating embeddings for {len(self.chunks)} chunks...")

        # Generate embeddings inline using OpenAI by default
        batch_size = 100  # OpenAI recommended batch size
        model = "text-embedding-3-large"

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

        # Debug: Check embedding dimensions
        if self.chunks and self.chunks[0].embedding:
            first_embedding = self.chunks[0].embedding
            print(f"🔍 First chunk embedding dimension: {len(first_embedding)}")
            print(
                f"🔍 Total chunks with embeddings: {len([c for c in self.chunks if c.embedding])}"
            )

        print("🏗️ Creating vector database...")

        try:
            # Create Chroma vector store with persistent storage
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

            # Create Chroma vector store with file-based naming
            db_name = get_database_name(properties_file, legal_file)
            chroma_persist_directory = f"databases/chroma_db_{db_name}"

            print(f"📁 Creating database: {chroma_persist_directory}")

            # CRITICAL: Create Chroma with proper metadata indexing
            print(f"   📊 Creating Chroma with {len(documents)} documents...")
            print(
                f"   🔍 Sample metadata fields: {list(documents[0].metadata.keys()) if documents else 'No documents'}"
            )

            # CRITICAL: Fix for ChromaDB 0.4.x compatibility
            try:
                self.vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_function,
                    persist_directory=chroma_persist_directory,
                )
            except Exception as e:
                print(f"   ⚠️ Standard Chroma creation failed: {e}")
                print(f"   🔧 Trying alternative creation method...")

                # Alternative method for ChromaDB 0.4.x
                import chromadb
                from chromadb.config import Settings

                # Create client with settings
                client = chromadb.PersistentClient(
                    path=chroma_persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )

                # Create collection
                collection = client.create_collection(
                    name="aspect_chunks", metadata={"hnsw:space": "cosine"}
                )

                # Add documents in batches
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i : i + batch_size]
                    ids = [
                        f"doc_{j}"
                        for j in range(i, min(i + batch_size, len(documents)))
                    ]
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]

                    collection.add(ids=ids, documents=texts, metadatas=metadatas)
                    print(
                        f"   📝 Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}"
                    )

                # Create LangChain wrapper
                from langchain_community.vectorstores import Chroma as LangChainChroma

                self.vector_db = LangChainChroma(
                    client=client,
                    collection_name="aspect_chunks",
                    embedding_function=embedding_function,
                )

            # CRITICAL: Ensure metadata is properly indexed
            print(f"   🔍 Verifying metadata indexing...")
            if self.vector_db:
                try:
                    # Test a simple metadata query to ensure indexing works
                    test_results = self.vector_db.get(
                        where={"type": "overview"}, limit=1
                    )
                    if (
                        test_results
                        and "documents" in test_results
                        and test_results["documents"]
                    ):
                        print(
                            f"   ✅ Metadata indexing verified - test query returned {len(test_results['documents'])} results"
                        )
                    else:
                        print(
                            f"   ⚠️ Metadata indexing may have issues - test query returned no results"
                        )
                except Exception as e:
                    print(f"   ⚠️ Metadata indexing test failed: {e}")
            else:
                print("   ⚠️ Vector database is None, cannot verify indexing")

            # Save the vector database metadata to disk for persistence
            self.save_vector_database(db_name, properties_file, legal_file)

            print(
                f"✅ Chroma vector database '{db_name}' created successfully with {len(documents)} chunks!"
            )
            return self.vector_db

        except Exception as e:
            print(f"❌ Error creating Chroma database: {e}")
            return None

    def save_vector_database(self, db_name: str, properties_file: str, legal_file: str):
        """Save the vector database metadata and chunks to disk for persistence"""
        try:
            # Create artifacts directory if it doesn't exist
            os.makedirs("databases", exist_ok=True)

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

            chunks_file = f"databases/aspect_chunks_{db_name}.json"
            with open(chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)

            # Save database metadata
            metadata = {
                "db_name": db_name,
                "properties_file": properties_file,
                "legal_file": legal_file,
                "chunk_count": len(chunks_data),
                "created_at": time.time(),
                "embedding_model": "text-embedding-3-large",
            }

            metadata_file = f"databases/db_metadata_{db_name}.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved {len(chunks_data)} chunks to {chunks_file}")
            print(f"💾 Saved database metadata to {metadata_file}")

        except Exception as e:
            print(f"⚠️ Warning: Could not save chunks data: {e}")

    def load_vector_database(
        self, properties_file: str, legal_file: str, embedding_model=None
    ) -> Optional[VectorStore]:
        """
        Load an existing vector database from disk if available

        Parameters:
            properties_file: Path to properties JSON file (for database naming)
            legal_file: Path to legal JSONL file (for database naming)
            embedding_model: The embedding model to use (ignored, uses OpenAI)

        Returns:
            VectorStore: The loaded vector database, or None if not available
        """
        try:
            # Generate database name based on files
            db_name = get_database_name(properties_file, legal_file)
            chroma_persist_directory = f"databases/chroma_db_{db_name}"

            if not os.path.exists(chroma_persist_directory):
                print(
                    f"📁 No database '{db_name}' found, will create new vector database"
                )
                return None

            # Check if files have been modified since database was created
            if check_files_modified(
                properties_file, legal_file, chroma_persist_directory
            ):
                print(
                    f"📝 Dataset files have been modified since database '{db_name}' was created"
                )
                print("🔄 Will recreate database with updated data")
                return None

            print(f"📁 Loading existing database '{db_name}' from disk...")

            # Create embedding function for loading
            from langchain_core.embeddings import Embeddings

            class OpenAIEmbeddingFunction(Embeddings):
                def __init__(self, openai_client):
                    self.openai_client = openai_client
                    self.embedding_dim = 3072  # Updated for text-embedding-3-large

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    # This should not be called when loading existing database
                    # but we need to implement it
                    return [[0.0] * self.embedding_dim for _ in texts]

                def embed_query(self, text: str) -> List[float]:
                    # For queries, we need to generate new embeddings with OpenAI
                    try:
                        response = self.openai_client.embeddings.create(
                            input=[text], model="text-embedding-3-large"
                        )
                        return response.data[0].embedding
                    except Exception as e:
                        print(f"❌ Error generating query embedding: {e}")
                        return [0.0] * self.embedding_dim

            embedding_function = OpenAIEmbeddingFunction(self.client)

            # Load existing Chroma database
            self.vector_db = Chroma(
                persist_directory=chroma_persist_directory,
                embedding_function=embedding_function,
            )

            print(f"✅ Successfully loaded existing database '{db_name}' from disk!")
            return self.vector_db

        except Exception as e:
            print(f"⚠️ Warning: Could not load Chroma database: {e}")
            return None


def create_aspect_based_vectordb(
    openai_api_key: str,
    properties_file: str = LOCAL_DATASET_PATH_LISTING_JSON,
    legal_file: str = LOCAL_DATASET_PATH_LEGAL_JSONL,
    embedding_model=None,
    force_recreate: bool = False,
):
    """
    Create an Aspect-Based vector database for real estate data with smart caching

    This function:
    1. Uses file-based naming for databases (e.g., run_ready_100, run_ready_904)
    2. Automatically loads existing database if files haven't changed
    3. Only recreates database if files are modified or force_recreate=True
    4. Uses OpenAI embeddings by default for optimal quality
    5. Saves everything to disk for future use

    Parameters:
        openai_api_key: OpenAI API key for embeddings
        properties_file: Path to properties JSON file (determines database name)
        legal_file: Path to legal JSONL file (determines database name)
        embedding_model: LangChain embedding model (optional, OpenAI used by default)
        force_recreate: If True, ignore existing database and create new one

    Returns:
        VectorStore: The created/loaded vector database
    """
    db_name = get_database_name(properties_file, legal_file)
    print(f"🚀 Creating/Loading Aspect-Based Vector Database: '{db_name}'")

    # Create default embedding model if none provided
    if embedding_model is None:
        try:
            from langchain_community.embeddings import OpenAIEmbeddings

            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large", api_key=openai_api_key
            )
            print("✅ Created default OpenAI embedding model")
        except Exception as e:
            print(f"❌ Failed to create default embedding model: {e}")
            return None

    # Initialize chunker
    chunker = AspectBasedChunker(openai_api_key)

    # Check if we should force recreation
    if force_recreate:
        print("🔄 Force recreate flag set - ignoring existing database")
    else:
        # First, try to load existing vector database from disk
        print(f"📁 Checking for existing database '{db_name}'...")
        existing_vectordb = chunker.load_vector_database(
            properties_file, legal_file, embedding_model
        )

        if existing_vectordb:
            print(f"✅ Successfully loaded existing database '{db_name}' from disk!")
            return existing_vectordb

    # If no existing database or force_recreate=True, create new one
    print(f"🆕 Creating new database '{db_name}'...")

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

    # Create vector database with file-based naming
    vectordb = chunker.create_vector_database(
        properties_file, legal_file, embedding_model
    )

    if vectordb:
        print(
            f"✅ Aspect-Based Vector Database '{db_name}' created and saved successfully!"
        )
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
