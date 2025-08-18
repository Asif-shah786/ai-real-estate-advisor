# 🧠 Chunking Strategy Comparison System - Usage Guide

## Overview

This system allows you to compare **5 different chunking strategies** for your Manchester Real Estate RAG application and see which one performs best for different types of questions.

## 🚀 Quick Start

### 1. Interactive Question Mode
```bash
python chunking_comparison_system.py
```
- Ask any question and see how all 5 strategies perform
- Compare scores, coverage, and response times
- Get detailed reports for each question

### 2. Batch Testing Mode
```bash
python batch_question_tester.py
```
- Automatically tests 24 predefined questions across 6 categories
- Provides comprehensive performance analysis
- Generates detailed reports and rankings

## 📊 Available Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **🏠 Property-Based** | Each property as a complete chunk | Property comparison, comprehensive queries |
| **🎯 Aspect-Based** | Separate chunks for crime, schools, transport | Specific aspect queries, focused searches |
| **🧠 Semantic-256** | Balanced chunks (max 256 words) | Balanced performance, consistent chunk sizes |
| **🧠 Semantic-512** | Medium chunks (max 512 words) | Good balance of context and specificity |
| **🧠 Semantic-1024** | Large chunks (max 1024 words) | Maximum context, fewer chunks |

## 🎯 Question Categories for Testing

### Crime & Safety
- "What are the crime rates in Manchester city center?"
- "Properties with low crime rates"
- "Safe neighborhoods in Manchester"

### Education & Schools
- "Properties near good schools in Greater Manchester"
- "Ofsted ratings for schools near properties"
- "Best school districts in Manchester"

### Transport & Connectivity
- "Transport links for properties under £300k"
- "Properties near train stations"
- "Good transport connections in Manchester"

### Legal & Requirements
- "Legal requirements for buying property in UK"
- "What legal documents do I need for property purchase?"
- "First time buyer legal advice Manchester"

### Property Features
- "Properties with low crime rates and good transport"
- "Family homes with good schools nearby"
- "Investment properties in Manchester"

### General Real Estate
- "Best areas to buy property in Manchester"
- "Property market trends in Greater Manchester"
- "Affordable housing options in Manchester"

## 🔍 How to Use Interactive Mode

1. **Start the system**: `python chunking_comparison_system.py`
2. **Wait for initialization**: System loads data and prepares all strategies
3. **Ask questions**: Type your real estate questions
4. **View results**: See how each strategy performs
5. **Get reports**: Automatic report generation and saving
6. **Continue or quit**: Ask more questions or exit

### Example Interactive Session:
```
❓ Your question: Properties near good schools in Greater Manchester

🔍 Asking question across all strategies: 'Properties near good schools in Greater Manchester'
================================================================================

📊 Testing Property-Based...
   📈 Score: 0.4523
   🎯 Coverage: 2 content types
   ⏱️  Response time: 0.234s
   🥇 Top result (Score: 0.4523):
      Type: property
      Content: Property Address: New Lane, Eccles M30...

📊 Testing Aspect-Based...
   📈 Score: 0.4872
   🎯 Coverage: 1 content types
   ⏱️  Response time: 0.198s
   🥇 Top result (Score: 0.4872):
      Type: schools
      Content: Property at New Lane, Eccles M30...

[continues for all strategies...]
```

## 📈 Understanding Results

### Score Metrics
- **Average Score**: Overall retrieval quality (0.0 to 1.0, higher is better)
- **Coverage**: Number of different content types in top results
- **Response Time**: How fast each strategy responds

### What Makes a Good Score
- **0.8+**: Excellent retrieval
- **0.6-0.8**: Very good retrieval
- **0.4-0.6**: Good retrieval
- **0.2-0.4**: Fair retrieval
- **0.0-0.2**: Poor retrieval

## 🏆 Previous Results

Based on our evaluation, **Aspect-Based chunking** typically performs best because:
- ✅ Highest average retrieval score (0.4872)
- ✅ Excellent for specific aspect queries
- ✅ Reduces noise in retrieval
- ✅ Better precision for focused questions

## 💡 Tips for Best Results

### 1. **Ask Specific Questions**
- ❌ "Tell me about Manchester properties"
- ✅ "Properties with low crime rates under £300k"

### 2. **Use Relevant Keywords**
- Include property features: "schools", "transport", "crime"
- Specify locations: "Manchester city center", "Greater Manchester"
- Mention property types: "family homes", "investment properties"

### 3. **Test Different Question Types**
- **Factual**: "What are the crime rates in..."
- **Comparative**: "Properties with better schools than..."
- **Specific**: "Legal requirements for first-time buyers"

## 📁 Generated Files

### Interactive Mode
- `comparison_report_[timestamp].md` - Individual question reports

### Batch Mode
- `batch_test_results_[timestamp].json` - Raw performance data
- `batch_test_report_[timestamp].md` - Comprehensive analysis

## 🔧 Technical Details

### Data Sources
- **Properties**: `dataset_v2/properties_with_crime_data.json`
- **Legal**: `dataset_v2/legal_uk_greater_manchester.jsonl`

### Embedding Model
- **Model**: `text-embedding-3-large` (3072 dimensions)
- **Provider**: OpenAI
- **Performance**: Optimized for real estate queries

### Vector Database
- **Type**: In-memory (numpy arrays)
- **Similarity**: Cosine similarity
- **Top-k**: Configurable (default: 5)

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   ❌ Error: OPENAI_API_KEY not found in environment variables!
   ```
   **Solution**: Create `.env` file with `OPENAI_API_KEY=your_key_here`

2. **Data Loading Error**
   ```
   ❌ Failed to load data files!
   ```
   **Solution**: Check file paths and ensure data files exist

3. **Memory Issues**
   ```
   ❌ Error during embedding generation
   ```
   **Solution**: Reduce batch size or use smaller embedding model

### Performance Optimization

- **Faster Testing**: Use batch mode for multiple questions
- **Memory Efficient**: Strategies are loaded once and reused
- **Caching**: Embeddings are generated once per strategy

## 📞 Support

If you encounter issues:
1. Check the error messages for specific details
2. Verify your `.env` file has the correct API key
3. Ensure all data files are in the correct locations
4. Check that all dependencies are installed

## 🎯 Next Steps

After finding the best chunking strategy:
1. **Integrate** the winning strategy into your main RAG app
2. **Fine-tune** chunk sizes and parameters
3. **Monitor** performance with real user queries
4. **Iterate** based on user feedback and performance metrics

---

**Happy Chunking! 🧠✨**
