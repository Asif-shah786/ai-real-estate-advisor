import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import openai
from dataclasses import dataclass
import hashlib
import time
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ChunkResult:
    """Data class to store chunking results and metadata"""

    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_type: str = "default"
    source_file: str = ""


class RealEstateRAGSystem:
    """
    Complete RAG system for Manchester Real Estate data with multiple chunking strategies
    """

    def __init__(self, openai_api_key: str):
        print("🔧 Initializing Real Estate RAG System...")
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.chunks = []
        self.embeddings = []
        self.chunk_metadata = []
        print("✅ RAG System initialized successfully!")

    def load_data(
        self, properties_file: str, legal_file: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Load the two JSON files containing property and legal data"""
        print(f"\n📁 Loading data files...")
        print(f"   Properties: {properties_file}")
        print(f"   Legal: {legal_file}")

        try:
            # Load properties data (JSON format)
            with open(properties_file, "r", encoding="utf-8") as f:
                properties_data = json.load(f)
            print(f"✅ Loaded {len(properties_data)} properties")

            # Load legal data (JSONL format)
            legal_data = []
            with open(legal_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            legal_item = json.loads(line)
                            legal_data.append(legal_item)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  Warning: Invalid JSON at line {line_num}: {e}")
                            continue
            print(f"✅ Loaded {len(legal_data)} legal entries")

            # Show sample data structure
            if properties_data:
                print(f"\n📊 Sample property fields: {list(properties_data[0].keys())}")
            if legal_data:
                print(f"📋 Sample legal fields: {list(legal_data[0].keys())}")

            return properties_data, legal_data

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return [], []

    def strategy_1_property_based_chunking(
        self, properties_data: List[Dict]
    ) -> List[ChunkResult]:
        """
        Strategy 1: Each property as a separate chunk with structured metadata
        Good for: Property-specific queries, comparisons
        """
        print(
            f"\n🏠 Strategy 1: Property-based chunking for {len(properties_data)} properties..."
        )
        chunks = []

        for i, prop in enumerate(properties_data):
            if i % 50 == 0:  # Progress indicator
                print(f"   Processing property {i+1}/{len(properties_data)}...")

            # Create comprehensive property description
            content_parts = []

            # Basic property info
            if "address" in prop:
                content_parts.append(f"Property Address: {prop['address']}")
            if "price_int" in prop:
                content_parts.append(f"price_int: £{prop['price_int']:,}")
            if "property_type" in prop:
                content_parts.append(f"Type: {prop['property_type']}")
            if "bedrooms" in prop:
                content_parts.append(f"Bedrooms: {prop['bedrooms']}")
            if "description" in prop:
                content_parts.append(
                    f"Description: {prop['description'][:500]}..."
                )  # Truncate long descriptions

            # Crime data
            if "crime_data" in prop and prop["crime_data"]:
                crime_info = prop["crime_data"]
                content_parts.append(
                    f"Crime Information: {json.dumps(crime_info, indent=2)}"
                )
            elif "crime_summary" in prop:
                content_parts.append(f"Crime Summary: {prop['crime_summary']}")

            # School data (if available)
            if "nearest_schools" in prop and prop["nearest_schools"]:
                content_parts.append(f"Nearby Schools: {prop['nearest_schools']}")

            # Transport data (if available)
            if "nearest_stations" in prop and prop["nearest_stations"]:
                content_parts.append(f"Transport Links: {prop['nearest_stations']}")

            content = "\n\n".join(content_parts)

            chunk = ChunkResult(
                chunk_id=f"property_{i}",
                content=content,
                metadata={
                    "type": "property",
                    "property_id": prop.get("property_id", i),
                    "address": prop.get("address", ""),
                    "price_int": prop.get("price_int", 0),
                    "postcode": prop.get("postcode", ""),
                    "has_crime_data": "crime_data" in prop or "crime_summary" in prop,
                    "has_schools": "nearest_schools" in prop
                    and bool(prop["nearest_schools"]),
                    "has_transport": "nearest_stations" in prop
                    and bool(prop["nearest_stations"]),
                },
                chunk_type="property_complete",
                source_file="properties",
            )
            chunks.append(chunk)

        print(f"✅ Created {len(chunks)} property chunks")
        return chunks

    def strategy_2_aspect_based_chunking(
        self, properties_data: List[Dict]
    ) -> List[ChunkResult]:
        """
        Strategy 2: Separate chunks for different aspects (crime, schools, transport)
        Good for: Specific aspect queries like "crime rates" or "school quality"
        """
        print(
            f"\n🎯 Strategy 2: Aspect-based chunking for {len(properties_data)} properties..."
        )
        chunks = []

        for i, prop in enumerate(properties_data):
            if i % 50 == 0:  # Progress indicator
                print(f"   Processing property {i+1}/{len(properties_data)}...")

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
                    ChunkResult(
                        chunk_id=f"property_{i}_crime",
                        content=crime_content,
                        metadata={
                            "type": "crime",
                            "property_id": prop.get("property_id", i),
                            "address": prop.get("address", ""),
                            "postcode": prop.get("postcode", ""),
                        },
                        chunk_type="aspect_crime",
                        source_file="properties",
                    )
                )

            # Schools chunk
            if "nearest_schools" in prop and prop["nearest_schools"]:
                schools_content = (
                    f"{base_info}\n\nNearby Schools:\n{prop['nearest_schools']}"
                )

                chunks.append(
                    ChunkResult(
                        chunk_id=f"property_{i}_schools",
                        content=schools_content,
                        metadata={
                            "type": "schools",
                            "property_id": prop.get("property_id", i),
                            "address": prop.get("address", ""),
                            "school_count": 1,  # Simplified for this data format
                        },
                        chunk_type="aspect_schools",
                        source_file="properties",
                    )
                )

            # Transport chunk
            if "nearest_stations" in prop and prop["nearest_stations"]:
                transport_content = (
                    f"{base_info}\n\nTransport Links:\n{prop['nearest_stations']}"
                )

                chunks.append(
                    ChunkResult(
                        chunk_id=f"property_{i}_transport",
                        content=transport_content,
                        metadata={
                            "type": "transport",
                            "property_id": prop.get("property_id", i),
                            "address": prop.get("address", ""),
                            "transport_count": 1,  # Simplified for this data format
                        },
                        chunk_type="aspect_transport",
                        source_file="properties",
                    )
                )

        print(f"✅ Created {len(chunks)} aspect-based chunks")
        return chunks

    def strategy_3_semantic_chunking(
        self, properties_data: List[Dict], max_chunk_size: int = 512
    ) -> List[ChunkResult]:
        """
        Strategy 3: Semantic chunking based on content similarity and size limits
        Good for: Balanced retrieval, avoiding oversized chunks
        """
        print(
            f"\n🧠 Strategy 3: Semantic chunking (max {max_chunk_size} words) for {len(properties_data)} properties..."
        )
        chunks = []
        current_chunk_content = []
        current_chunk_metadata = []
        current_size = 0
        chunk_counter = 0

        for i, prop in enumerate(properties_data):
            if i % 50 == 0:  # Progress indicator
                print(f"   Processing property {i+1}/{len(properties_data)}...")

            # Create property text representation
            prop_text = self._create_property_text(prop)
            prop_size = len(prop_text.split())

            # If adding this property would exceed chunk size, finalize current chunk
            if current_size + prop_size > max_chunk_size and current_chunk_content:
                chunk = self._finalize_semantic_chunk(
                    current_chunk_content, current_chunk_metadata, chunk_counter
                )
                chunks.append(chunk)

                # Reset for new chunk
                current_chunk_content = []
                current_chunk_metadata = []
                current_size = 0
                chunk_counter += 1

            # Add property to current chunk
            current_chunk_content.append(prop_text)
            current_chunk_metadata.append(
                {
                    "property_id": prop.get("property_id", i),
                    "address": prop.get("address", ""),
                    "postcode": prop.get("postcode", ""),
                }
            )
            current_size += prop_size

        # Don't forget the last chunk
        if current_chunk_content:
            chunk = self._finalize_semantic_chunk(
                current_chunk_content, current_chunk_metadata, chunk_counter
            )
            chunks.append(chunk)

        print(f"✅ Created {len(chunks)} semantic chunks")
        return chunks

    def strategy_4_legal_qa_chunking(self, legal_data: List[Dict]) -> List[ChunkResult]:
        """
        Strategy 4: Legal Q&A pairs as individual chunks
        Good for: Direct legal question answering
        """
        print(
            f"\n⚖️  Strategy 4: Legal Q&A chunking for {len(legal_data)} legal entries..."
        )
        chunks = []

        for i, legal_item in enumerate(legal_data):
            if i % 10 == 0:  # Progress indicator
                print(f"   Processing legal entry {i+1}/{len(legal_data)}...")

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
                ChunkResult(
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
                    },
                    chunk_type="legal_info",
                    source_file="legal",
                )
            )

        print(f"✅ Created {len(chunks)} legal chunks")
        return chunks

    def _create_property_text(self, prop: Dict) -> str:
        """Helper to create standardized property text"""
        parts = []

        # Basic info
        if "address" in prop:
            parts.append(f"Address: {prop['address']}")
        if "price_int" in prop:
            parts.append(f"price_int: £{prop['price_int']:,}")

        # Crime summary
        if "crime_data" in prop and prop["crime_data"]:
            parts.append(f"Crime data available: {bool(prop['crime_data'])}")
        elif "crime_summary" in prop:
            parts.append(f"Crime: {prop['crime_summary']}")

        # Schools summary
        if "nearest_schools" in prop and prop["nearest_schools"]:
            parts.append(f"Schools: {prop['nearest_schools']}")

        # Transport summary
        if "nearest_stations" in prop and prop["nearest_stations"]:
            parts.append(f"Transport: {prop['nearest_stations']}")

        return ". ".join(parts)

    def _finalize_semantic_chunk(
        self, content_list: List[str], metadata_list: List[Dict], chunk_id: int
    ) -> ChunkResult:
        """Helper to create semantic chunk from accumulated content"""
        content = "\n\n---\n\n".join(content_list)

        return ChunkResult(
            chunk_id=f"semantic_{chunk_id}",
            content=content,
            metadata={
                "type": "semantic_group",
                "property_count": len(content_list),
                "properties": metadata_list,
            },
            chunk_type="semantic_group",
            source_file="properties",
        )

    def generate_embeddings(
        self, chunks: List[ChunkResult], model: str = "text-embedding-3-large"
    ) -> List[ChunkResult]:
        """
        Generate embeddings for chunks using OpenAI's embedding models
        """
        print(f"\n🧠 Generating embeddings for {len(chunks)} chunks using {model}...")

        batch_size = 100  # OpenAI recommended batch size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]

            print(
                f"   Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)..."
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

        print(f"✅ Embedding generation completed for {len(chunks)} chunks")
        return chunks

    def evaluate_chunking_strategy(
        self, chunks: List[ChunkResult], test_queries: List[str], strategy_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a chunking strategy using test queries
        """
        print(f"\n📊 Evaluating {strategy_name}...")

        # Generate embeddings if not already done
        if not chunks[0].embedding:
            print("   Generating embeddings for evaluation...")
            chunks_with_embeddings = self.generate_embeddings(chunks)
        else:
            chunks_with_embeddings = chunks

        results = {
            "strategy_name": strategy_name,
            "total_chunks": len(chunks),
            "avg_chunk_size": np.mean([len(chunk.content.split()) for chunk in chunks]),
            "chunk_size_std": np.std([len(chunk.content.split()) for chunk in chunks]),
            "retrieval_scores": [],
            "coverage_scores": [],
        }

        print(f"   Testing with {len(test_queries)} queries...")

        # Test retrieval quality
        for i, query in enumerate(test_queries):
            print(f"   Query {i+1}/{len(test_queries)}: {query[:50]}...")

            try:
                query_embedding = (
                    self.client.embeddings.create(
                        input=[query], model="text-embedding-3-large"
                    )
                    .data[0]
                    .embedding
                )

                # Calculate similarities
                similarities = []
                for chunk in chunks_with_embeddings:
                    if chunk.embedding:
                        # Convert to numpy arrays for cosine_similarity
                        query_array = np.array(query_embedding).reshape(1, -1)
                        chunk_array = np.array(chunk.embedding).reshape(1, -1)
                        sim = cosine_similarity(query_array, chunk_array)[0][0]
                        similarities.append(sim)
                    else:
                        similarities.append(0.0)

                # Get top 5 results
                top_indices = np.argsort(similarities)[-5:][::-1]
                top_scores = [similarities[idx] for idx in top_indices]
                results["retrieval_scores"].extend(top_scores)

                # Coverage: How many different types of content in top results
                top_types = set(
                    [
                        chunks_with_embeddings[idx].metadata.get("type", "unknown")
                        for idx in top_indices
                    ]
                )
                results["coverage_scores"].append(len(top_types))

                print(f"     ✅ Query {i+1} completed")

            except Exception as e:
                print(f"     ❌ Error processing query {i+1}: {e}")
                results["retrieval_scores"].extend([0.0] * 5)
                results["coverage_scores"].append(0)

        results["avg_retrieval_score"] = np.mean(results["retrieval_scores"])
        results["avg_coverage"] = np.mean(results["coverage_scores"])

        print(f"   ✅ Evaluation completed for {strategy_name}")
        return results

    def run_comprehensive_evaluation(
        self, properties_data: List[Dict], legal_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Run all chunking strategies and evaluate them
        """
        # Define test queries for evaluation
        test_queries = [
            "What are the crime rates in Manchester city center?",
            "Properties near good schools in Greater Manchester",
            "Legal requirements for buying property in UK",
            "Transport links for properties under £300k",
            "Ofsted ratings for schools near properties",
            "What legal documents do I need for property purchase?",
            "Properties with low crime rates and good transport",
            "First time buyer legal advice Manchester",
        ]

        print("\n🚀 Starting comprehensive evaluation of chunking strategies...")
        print(f"   Test queries: {len(test_queries)}")

        strategies_results = {}

        # Strategy 1: Property-based chunking
        print("\n" + "=" * 60)
        strategy1_chunks = self.strategy_1_property_based_chunking(properties_data)
        legal_chunks = self.strategy_4_legal_qa_chunking(legal_data)
        combined_chunks1 = strategy1_chunks + legal_chunks

        strategies_results["Property-Based"] = self.evaluate_chunking_strategy(
            combined_chunks1, test_queries, "Property-Based Chunking"
        )

        # Strategy 2: Aspect-based chunking
        print("\n" + "=" * 60)
        strategy2_chunks = self.strategy_2_aspect_based_chunking(properties_data)
        combined_chunks2 = strategy2_chunks + legal_chunks

        strategies_results["Aspect-Based"] = self.evaluate_chunking_strategy(
            combined_chunks2, test_queries, "Aspect-Based Chunking"
        )

        # Strategy 3: Semantic chunking (different sizes)
        for chunk_size in [256, 512, 1024]:
            print("\n" + "=" * 60)
            strategy3_chunks = self.strategy_3_semantic_chunking(
                properties_data, chunk_size
            )
            combined_chunks3 = strategy3_chunks + legal_chunks

            strategies_results[f"Semantic-{chunk_size}"] = (
                self.evaluate_chunking_strategy(
                    combined_chunks3,
                    test_queries,
                    f"Semantic Chunking ({chunk_size} words)",
                )
            )

        print("\n" + "=" * 60)
        print("🎉 Comprehensive evaluation completed!")
        return strategies_results

    def create_vector_database(self, chunks: List[ChunkResult]) -> Dict[str, Any]:
        """
        Create a simple in-memory vector database
        In production, you'd use Pinecone, Weaviate, or Chroma
        """
        print(f"\n🗄️  Creating vector database with {len(chunks)} chunks...")

        # Filter chunks with embeddings
        chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding]
        print(f"   Chunks with embeddings: {len(chunks_with_embeddings)}")

        if not chunks_with_embeddings:
            print("   ❌ No chunks with embeddings found!")
            return {}

        embeddings_matrix = np.array(
            [chunk.embedding for chunk in chunks_with_embeddings]
        )

        vector_db = {
            "embeddings": embeddings_matrix,
            "chunks": chunks_with_embeddings,
            "metadata": [chunk.metadata for chunk in chunks_with_embeddings],
            "total_chunks": len(chunks_with_embeddings),
        }

        print(f"✅ Vector database created successfully!")
        return vector_db

    def search_similar(
        self, query: str, vector_db: Dict[str, Any], top_k: int = 5
    ) -> List[Tuple[ChunkResult, float]]:
        """
        Search for similar chunks in the vector database
        """
        if not vector_db or "embeddings" not in vector_db:
            print("❌ Vector database is empty or invalid!")
            return []

        print(f"\n🔍 Searching for: {query}")

        try:
            # Generate query embedding
            query_embedding = (
                self.client.embeddings.create(
                    input=[query], model="text-embedding-3-large"
                )
                .data[0]
                .embedding
            )

            # Calculate similarities
            query_array = np.array(query_embedding).reshape(1, -1)
            similarities = cosine_similarity(query_array, vector_db["embeddings"])[0]

            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            results = []
            for idx in top_indices:
                chunk = vector_db["chunks"][idx]
                score = similarities[idx]
                results.append((chunk, score))

            print(f"✅ Found {len(results)} results")
            return results

        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report
        """
        print("\n📝 Generating evaluation report...")

        report = "# Manchester Real Estate RAG System - Evaluation Report\n\n"

        # Sort strategies by performance
        sorted_strategies = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]["avg_retrieval_score"],
            reverse=True,
        )

        report += "## Strategy Performance Rankings\n\n"

        for rank, (strategy_name, results) in enumerate(sorted_strategies, 1):
            report += f"### {rank}. {strategy_name}\n"
            report += (
                f"- **Average Retrieval Score**: {results['avg_retrieval_score']:.4f}\n"
            )
            report += f"- **Average Coverage**: {results['avg_coverage']:.2f}\n"
            report += f"- **Total Chunks**: {results['total_chunks']}\n"
            report += (
                f"- **Average Chunk Size**: {results['avg_chunk_size']:.1f} words\n"
            )
            report += f"- **Chunk Size Std Dev**: {results['chunk_size_std']:.1f}\n\n"

        # Recommendations
        best_strategy = sorted_strategies[0]
        report += "## Recommendations\n\n"
        report += f"**Best Overall Strategy**: {best_strategy[0]}\n\n"

        report += "### Why this strategy works best:\n"
        if "Property-Based" in best_strategy[0]:
            report += "- Maintains complete property context\n"
            report += "- Good for property comparison queries\n"
            report += (
                "- Preserves relationships between crime, schools, and transport data\n"
            )
        elif "Aspect-Based" in best_strategy[0]:
            report += (
                "- Excellent for specific aspect queries (crime, schools, transport)\n"
            )
            report += "- Reduces noise in retrieval\n"
            report += "- Better precision for focused questions\n"
        elif "Semantic" in best_strategy[0]:
            report += "- Balanced approach between context and specificity\n"
            report += "- Consistent chunk sizes for stable performance\n"
            report += "- Good scalability for large datasets\n"

        report += "\n### OpenAI Embedding Model Recommendation:\n"
        report += "- **Primary**: `text-embedding-3-large` (3072 dimensions)\n"
        report += "  - Best performance for complex real estate queries\n"
        report += "  - Superior semantic understanding\n"
        report += "- **Alternative**: `text-embedding-3-small` (1536 dimensions)\n"
        report += "  - Cost-effective option with good performance\n"
        report += "  - Faster processing for large datasets\n"

        print("✅ Evaluation report generated!")
        return report


# Usage and demonstration
if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables!")
        print("   Please set your OpenAI API key in the .env file")
        exit(1)

    print("🚀 Starting Real Estate RAG System Evaluation...")
    print(
        f"   API Key: {'*' * (len(api_key) - 8) + api_key[-8:] if len(api_key) > 8 else '***'}"
    )

    # Initialize system
    rag_system = RealEstateRAGSystem(api_key)

    # Load your data
    properties_data, legal_data = rag_system.load_data(
        "dataset_v2/properties_with_crime_data.json",
        "dataset_v2/legal_uk_greater_manchester.jsonl",
    )

    if not properties_data or not legal_data:
        print("❌ Failed to load data files!")
        exit(1)

    # Run comprehensive evaluation
    evaluation_results = rag_system.run_comprehensive_evaluation(
        properties_data, legal_data
    )

    # Generate report
    report = rag_system.generate_evaluation_report(evaluation_results)

    # Save report to file
    with open("chunking_evaluation_report.md", "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("📊 EVALUATION COMPLETE!")
    print("📄 Report saved to: chunking_evaluation_report.md")
    print("\n" + report)

    # Use the best strategy to create your final vector database
    print("\n🔧 Creating production vector database with best strategy...")
    best_chunks = rag_system.strategy_1_property_based_chunking(properties_data)
    legal_chunks = rag_system.strategy_4_legal_qa_chunking(legal_data)
    all_chunks = best_chunks + legal_chunks

    # Generate embeddings and create vector DB
    chunks_with_embeddings = rag_system.generate_embeddings(all_chunks)
    vector_db = rag_system.create_vector_database(chunks_with_embeddings)

    # Test search
    if vector_db:
        print("\n🧪 Testing search functionality...")
        results = rag_system.search_similar(
            "Properties with good schools and low crime", vector_db, top_k=3
        )

        print(f"\n🔍 Search Results:")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {score:.4f}) ---")
            print(f"Type: {chunk.metadata.get('type', 'unknown')}")
            print(f"Content: {chunk.content[:200]}...")

    print("\n🎉 All done! Check the evaluation report for detailed results.")


# Demo function to show expected usage
def demo_usage():
    """
    Demonstration of how to use the system
    """
    print(
        """
    # Example Usage:
    
    # 1. Initialize system
    rag_system = RealEstateRAGSystem("your-openai-api-key")
    
    # 2. Load your data files
    properties_data, legal_data = rag_system.load_data(
        "dataset_v2/properties_with_crime_data.json", 
        "dataset_v2/legal_uk_greater_manchester.jsonl"
    )
    
    # 3. Run evaluation to find best strategy
    evaluation_results = rag_system.run_comprehensive_evaluation(properties_data, legal_data)
    
    # 4. Generate and review report
    report = rag_system.generate_evaluation_report(evaluation_results)
    print(report)
    
    # 5. Implement best strategy
    # (Based on evaluation results)
    
    # 6. Create production vector database
    # vector_db = rag_system.create_vector_database(best_chunks)
    """
    )
