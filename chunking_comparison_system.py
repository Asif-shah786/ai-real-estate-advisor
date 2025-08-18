import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import openai
from dataclasses import dataclass
import time
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from best_chunking_strategy import RealEstateRAGSystem, ChunkResult

# Load environment variables
load_dotenv()

@dataclass
class StrategyResult:
    """Data class to store strategy comparison results"""
    strategy_name: str
    top_chunks: List[Tuple[ChunkResult, float]]
    avg_score: float
    coverage: int
    response_time: float

class ChunkingComparisonSystem:
    """
    System to compare different chunking strategies for real estate queries
    """
    
    def __init__(self, openai_api_key: str):
        print("🔧 Initializing Chunking Comparison System...")
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.rag_system = RealEstateRAGSystem(openai_api_key)
        self.strategies = {}
        self.vector_dbs = {}
        self.properties_data = []
        self.legal_data = []
        print("✅ Comparison System initialized successfully!")
    
    def load_and_prepare_data(self):
        """Load data and prepare all chunking strategies"""
        print("\n📁 Loading and preparing data for all strategies...")
        
        # Load data
        self.properties_data, self.legal_data = self.rag_system.load_data(
            "dataset_v2/properties_with_crime_data.json",
            "dataset_v2/legal_uk_greater_manchester.jsonl"
        )
        
        if not self.properties_data or not self.legal_data:
            print("❌ Failed to load data files!")
            return False
        
        print(f"✅ Data loaded: {len(self.properties_data)} properties, {len(self.legal_data)} legal entries")
        
        # Prepare all strategies
        self._prepare_all_strategies()
        
        return True
    
    def _prepare_all_strategies(self):
        """Prepare all chunking strategies with embeddings"""
        print("\n🏗️  Preparing all chunking strategies...")
        
        # Strategy 1: Property-based
        print("\n🏠 Preparing Property-Based Strategy...")
        property_chunks = self.rag_system.strategy_1_property_based_chunking(self.properties_data)
        legal_chunks = self.rag_system.strategy_4_legal_qa_chunking(self.legal_data)
        combined_chunks = property_chunks + legal_chunks
        
        # Generate embeddings
        chunks_with_embeddings = self.rag_system.generate_embeddings(combined_chunks)
        vector_db = self.rag_system.create_vector_database(chunks_with_embeddings)
        
        self.strategies['Property-Based'] = {
            'chunks': chunks_with_embeddings,
            'vector_db': vector_db,
            'description': 'Each property as a complete chunk with all information'
        }
        
        # Strategy 2: Aspect-based
        print("\n🎯 Preparing Aspect-Based Strategy...")
        aspect_chunks = self.rag_system.strategy_2_aspect_based_chunking(self.properties_data)
        combined_aspect_chunks = aspect_chunks + legal_chunks
        
        chunks_with_embeddings = self.rag_system.generate_embeddings(combined_aspect_chunks)
        vector_db = self.rag_system.create_vector_database(chunks_with_embeddings)
        
        self.strategies['Aspect-Based'] = {
            'chunks': chunks_with_embeddings,
            'vector_db': vector_db,
            'description': 'Separate chunks for crime, schools, transport aspects'
        }
        
        # Strategy 3: Semantic chunking variants
        for chunk_size in [256, 512, 1024]:
            print(f"\n🧠 Preparing Semantic-{chunk_size} Strategy...")
            semantic_chunks = self.rag_system.strategy_3_semantic_chunking(self.properties_data, chunk_size)
            combined_semantic_chunks = semantic_chunks + legal_chunks
            
            chunks_with_embeddings = self.rag_system.generate_embeddings(combined_semantic_chunks)
            vector_db = self.rag_system.create_vector_database(chunks_with_embeddings)
            
            self.strategies[f'Semantic-{chunk_size}'] = {
                'chunks': chunks_with_embeddings,
                'vector_db': vector_db,
                'description': f'Semantic chunking with max {chunk_size} words per chunk'
            }
        
        print(f"\n✅ All {len(self.strategies)} strategies prepared successfully!")
        
        # Show strategy summary
        for name, strategy in self.strategies.items():
            chunk_count = len(strategy['chunks'])
            print(f"   {name}: {chunk_count} chunks - {strategy['description']}")
    
    def ask_question_all_strategies(self, question: str, top_k: int = 5) -> Dict[str, StrategyResult]:
        """
        Ask the same question to all chunking strategies and compare results
        """
        print(f"\n🔍 Asking question across all strategies: '{question}'")
        print("=" * 80)
        
        results = {}
        
        for strategy_name, strategy_data in self.strategies.items():
            print(f"\n📊 Testing {strategy_name}...")
            
            start_time = time.time()
            
            # Search in this strategy's vector database
            search_results = self._search_in_strategy(question, strategy_data['vector_db'], top_k)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Calculate metrics
            if search_results:
                scores = [score for _, score in search_results]
                avg_score = float(np.mean(scores))
                coverage = len(set([chunk.metadata.get('type', 'unknown') for chunk, _ in search_results]))
            else:
                avg_score = 0.0
                coverage = 0
            
            # Store results
            results[strategy_name] = StrategyResult(
                strategy_name=strategy_name,
                top_chunks=search_results,
                avg_score=avg_score,
                coverage=coverage,
                response_time=response_time
            )
            
            # Display results
            print(f"   📈 Score: {avg_score:.4f}")
            print(f"   🎯 Coverage: {coverage} content types")
            print(f"   ⏱️  Response time: {response_time:.3f}s")
            
            # Show top result preview
            if search_results:
                top_chunk, top_score = search_results[0]
                print(f"   🥇 Top result (Score: {top_score:.4f}):")
                print(f"      Type: {top_chunk.metadata.get('type', 'unknown')}")
                print(f"      Content: {top_chunk.content[:100]}...")
        
        return results
    
    def _search_in_strategy(self, query: str, vector_db: Dict[str, Any], top_k: int) -> List[Tuple[ChunkResult, float]]:
        """Search within a specific strategy's vector database"""
        if not vector_db or 'embeddings' not in vector_db:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            ).data[0].embedding
            
            # Calculate similarities
            query_array = np.array(query_embedding).reshape(1, -1)
            similarities = cosine_similarity(query_array, vector_db['embeddings'])[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                chunk = vector_db['chunks'][idx]
                score = similarities[idx]
                results.append((chunk, score))
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error searching in strategy: {e}")
            return []
    
    def generate_comparison_report(self, question: str, results: Dict[str, StrategyResult]) -> str:
        """Generate a detailed comparison report"""
        print("\n📝 Generating comparison report...")
        
        # Sort strategies by average score
        sorted_strategies = sorted(
            results.items(),
            key=lambda x: x[1].avg_score,
            reverse=True
        )
        
        report = f"# Chunking Strategy Comparison Report\n\n"
        report += f"**Question**: {question}\n\n"
        
        report += "## Strategy Performance Rankings\n\n"
        
        for rank, (strategy_name, result) in enumerate(sorted_strategies, 1):
            report += f"### {rank}. {strategy_name}\n"
            report += f"- **Average Score**: {result.avg_score:.4f}\n"
            report += f"- **Coverage**: {result.coverage} content types\n"
            report += f"- **Response Time**: {result.response_time:.3f}s\n"
            report += f"- **Top Result Score**: {result.top_chunks[0][1]:.4f if result.top_chunks else 0:.4f}\n\n"
        
        # Winner analysis
        winner = sorted_strategies[0]
        report += "## 🏆 Winner Analysis\n\n"
        report += f"**Best Strategy**: {winner[0]}\n\n"
        report += f"**Why it won**:\n"
        report += f"- Highest average retrieval score: {winner[1].avg_score:.4f}\n"
        report += f"- Good content coverage: {winner[1].coverage} types\n"
        report += f"- Fast response time: {winner[1].response_time:.3f}s\n\n"
        
        # Detailed results for winner
        report += f"**Top Results from {winner[0]}**:\n\n"
        for i, (chunk, score) in enumerate(winner[1].top_chunks[:3], 1):
            report += f"**Result {i}** (Score: {score:.4f})\n"
            report += f"- Type: {chunk.metadata.get('type', 'unknown')}\n"
            report += f"- Content: {chunk.content[:200]}...\n\n"
        
        return report
    
    def interactive_question_mode(self):
        """Interactive mode for asking questions and comparing strategies"""
        print("\n🎯 Interactive Question Mode")
        print("=" * 50)
        print("Ask questions to compare chunking strategies!")
        print("Type 'quit' to exit, 'help' for example questions")
        print("=" * 50)
        
        example_questions = [
            "What are the crime rates in Manchester city center?",
            "Properties near good schools in Greater Manchester",
            "Legal requirements for buying property in UK",
            "Transport links for properties under £300k",
            "Properties with low crime rates and good transport",
            "First time buyer legal advice Manchester",
            "What legal documents do I need for property purchase?",
            "Properties with good Ofsted ratings nearby"
        ]
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'help':
                    print("\n📚 Example Questions:")
                    for i, q in enumerate(example_questions, 1):
                        print(f"   {i}. {q}")
                    continue
                elif not question:
                    continue
                
                # Ask question to all strategies
                results = self.ask_question_all_strategies(question)
                
                # Generate and display report
                report = self.generate_comparison_report(question, results)
                
                print("\n" + "=" * 80)
                print("📊 COMPARISON REPORT")
                print("=" * 80)
                print(report)
                
                # Save report
                timestamp = int(time.time())
                filename = f"comparison_report_{timestamp}.md"
                with open(filename, "w") as f:
                    f.write(report)
                print(f"\n📄 Report saved to: {filename}")
                
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Ask another question? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    print("👋 Thanks for using the comparison system!")
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit")

def main():
    """Main function to run the comparison system"""
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables!")
        print("   Please set your OpenAI API key in the .env file")
        return
    
    print("🚀 Starting Chunking Strategy Comparison System...")
    
    # Initialize system
    comparison_system = ChunkingComparisonSystem(api_key)
    
    # Load and prepare data
    if not comparison_system.load_and_prepare_data():
        print("❌ Failed to prepare data. Exiting.")
        return
    
    # Start interactive mode
    comparison_system.interactive_question_mode()

if __name__ == "__main__":
    main()
