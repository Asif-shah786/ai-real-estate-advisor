import json
import time
from typing import List, Dict, Any
from chunking_comparison_system import ChunkingComparisonSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_batch_test():
    """Run batch testing with predefined questions"""
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables!")
        print("   Please set your OpenAI API key in the .env file")
        return
    
    print("🚀 Starting Batch Question Testing...")
    
    # Initialize system
    comparison_system = ChunkingComparisonSystem(api_key)
    
    # Load and prepare data
    if not comparison_system.load_and_prepare_data():
        print("❌ Failed to prepare data. Exiting.")
        return
    
    # Predefined test questions organized by category
    test_questions = {
        "Crime & Safety": [
            "What are the crime rates in Manchester city center?",
            "Properties with low crime rates",
            "Crime statistics for Greater Manchester areas",
            "Safe neighborhoods in Manchester"
        ],
        "Education & Schools": [
            "Properties near good schools in Greater Manchester",
            "Ofsted ratings for schools near properties",
            "Best school districts in Manchester",
            "Properties with excellent school access"
        ],
        "Transport & Connectivity": [
            "Transport links for properties under £300k",
            "Properties near train stations",
            "Good transport connections in Manchester",
            "Properties with easy access to motorways"
        ],
        "Legal & Requirements": [
            "Legal requirements for buying property in UK",
            "What legal documents do I need for property purchase?",
            "First time buyer legal advice Manchester",
            "Property purchase legal checklist"
        ],
        "Property Features": [
            "Properties with low crime rates and good transport",
            "Family homes with good schools nearby",
            "Investment properties in Manchester",
            "Properties with garden and parking"
        ],
        "General Real Estate": [
            "Best areas to buy property in Manchester",
            "Property market trends in Greater Manchester",
            "Affordable housing options in Manchester",
            "New developments in Manchester area"
        ]
    }
    
    print(f"\n📋 Testing {sum(len(questions) for questions in test_questions.values())} questions across {len(test_questions)} categories...")
    
    # Results storage
    all_results = {}
    strategy_performance = {}
    
    # Initialize strategy performance tracking
    for strategy_name in comparison_system.strategies.keys():
        strategy_performance[strategy_name] = {
            'total_score': 0.0,
            'total_questions': 0,
            'category_wins': {},
            'avg_response_time': 0.0,
            'total_response_time': 0.0
        }
    
    # Test each category
    for category, questions in test_questions.items():
        print(f"\n{'='*60}")
        print(f"🏷️  Testing Category: {category}")
        print(f"{'='*60}")
        
        category_results = {}
        
        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}/{len(questions)}: {question}")
            
            # Test question across all strategies
            results = comparison_system.ask_question_all_strategies(question, top_k=3)
            
            # Store results
            category_results[question] = results
            
            # Update strategy performance
            for strategy_name, result in results.items():
                strategy_performance[strategy_name]['total_score'] += result.avg_score
                strategy_performance[strategy_name]['total_questions'] += 1
                strategy_performance[strategy_name]['total_response_time'] += result.response_time
                
                # Track category wins
                if strategy_name not in strategy_performance[strategy_name]['category_wins']:
                    strategy_performance[strategy_name]['category_wins'][category] = 0
                
                # Check if this strategy won for this question
                if result.avg_score == max([r.avg_score for r in results.values()]):
                    strategy_performance[strategy_name]['category_wins'][category] += 1
        
        all_results[category] = category_results
        
        # Show category summary
        print(f"\n📊 Category Summary for {category}:")
        category_scores = {}
        for strategy_name, result in results.items():
            category_scores[strategy_name] = result.avg_score
        
        sorted_strategies = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (strategy, score) in enumerate(sorted_strategies, 1):
            print(f"   {rank}. {strategy}: {score:.4f}")
    
    # Calculate final performance metrics
    print(f"\n{'='*80}")
    print("🏆 FINAL PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    for strategy_name, performance in strategy_performance.items():
        if performance['total_questions'] > 0:
            performance['avg_score'] = performance['total_score'] / performance['total_questions']
            performance['avg_response_time'] = performance['total_response_time'] / performance['total_questions']
    
    # Sort strategies by average score
    sorted_strategies = sorted(
        strategy_performance.items(),
        key=lambda x: x[1]['avg_score'],
        reverse=True
    )
    
    print("\n## Overall Strategy Rankings\n")
    
    for rank, (strategy_name, performance) in enumerate(sorted_strategies, 1):
        print(f"### {rank}. {strategy_name}")
        print(f"- **Average Score**: {performance['avg_score']:.4f}")
        print(f"- **Questions Tested**: {performance['total_questions']}")
        print(f"- **Average Response Time**: {performance['avg_response_time']:.3f}s")
        
        # Show category wins
        if performance['category_wins']:
            print(f"- **Category Wins**:")
            for category, wins in performance['category_wins'].items():
                if wins > 0:
                    print(f"  - {category}: {wins} wins")
        print()
    
    # Winner analysis
    winner = sorted_strategies[0]
    print("## 🏆 Overall Winner\n")
    print(f"**Best Strategy**: {winner[0]}")
    print(f"**Average Score**: {winner[1]['avg_score']:.4f}")
    print(f"**Performance**: {winner[1]['avg_score']:.1%} better than average")
    
    # Save comprehensive results
    timestamp = int(time.time())
    results_filename = f"batch_test_results_{timestamp}.json"
    
    # Prepare results for JSON serialization
    serializable_results = {}
    for category, questions in all_results.items():
        serializable_results[category] = {}
        for question, strategies in questions.items():
            serializable_results[category][question] = {}
            for strategy_name, result in strategies.items():
                serializable_results[category][question][strategy_name] = {
                    'avg_score': result.avg_score,
                    'coverage': result.coverage,
                    'response_time': result.response_time,
                    'top_chunk_type': result.top_chunks[0][0].metadata.get('type', 'unknown') if result.top_chunks else 'none',
                    'top_chunk_score': result.top_chunks[0][1] if result.top_chunks else 0.0
                }
    
    # Add performance summary
    serializable_results['performance_summary'] = {}
    for strategy_name, performance in strategy_performance.items():
        serializable_results['performance_summary'][strategy_name] = {
            'avg_score': performance.get('avg_score', 0.0),
            'total_questions': performance['total_questions'],
            'avg_response_time': performance.get('avg_response_time', 0.0),
            'category_wins': performance['category_wins']
        }
    
    with open(results_filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Comprehensive results saved to: {results_filename}")
    
    # Generate markdown report
    report_filename = f"batch_test_report_{timestamp}.md"
    generate_markdown_report(all_results, strategy_performance, report_filename)
    
    print(f"📝 Markdown report saved to: {report_filename}")
    print("\n🎉 Batch testing completed!")

def generate_markdown_report(all_results: Dict, strategy_performance: Dict, filename: str):
    """Generate a comprehensive markdown report"""
    
    report = "# Batch Question Testing Report\n\n"
    report += "## Overview\n\n"
    report += f"**Total Questions Tested**: {sum(len(questions) for questions in all_results.values())}\n"
    report += f"**Categories**: {len(all_results)}\n"
    report += f"**Strategies**: {len(strategy_performance)}\n\n"
    
    # Overall rankings
    sorted_strategies = sorted(
        strategy_performance.items(),
        key=lambda x: x[1].get('avg_score', 0.0),
        reverse=True
    )
    
    report += "## Overall Strategy Rankings\n\n"
    
    for rank, (strategy_name, performance) in enumerate(sorted_strategies, 1):
        report += f"### {rank}. {strategy_name}\n"
        report += f"- **Average Score**: {performance.get('avg_score', 0.0):.4f}\n"
        report += f"- **Questions Tested**: {performance['total_questions']}\n"
        report += f"- **Average Response Time**: {performance.get('avg_response_time', 0.0):.3f}s\n\n"
    
    # Category breakdown
    report += "## Category Performance Breakdown\n\n"
    
    for category, questions in all_results.items():
        report += f"### {category}\n\n"
        
        for question, strategies in questions.items():
            report += f"**Q**: {question}\n\n"
            
            # Sort strategies by score for this question
            question_scores = [(name, result.avg_score) for name, result in strategies.items()]
            question_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (strategy_name, score) in enumerate(question_scores, 1):
                report += f"{rank}. {strategy_name}: {score:.4f}\n"
            
            report += "\n"
    
    # Save report
    with open(filename, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    run_batch_test()
