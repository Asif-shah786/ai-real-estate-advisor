# Manchester Real Estate RAG System - Evaluation Report

## Strategy Performance Rankings

### 1. Aspect-Based
- **Average Retrieval Score**: 0.4872
- **Average Coverage**: 1.00
- **Total Chunks**: 48
- **Average Chunk Size**: 43.4 words
- **Chunk Size Std Dev**: 11.9

### 2. Property-Based
- **Average Retrieval Score**: 0.4377
- **Average Coverage**: 1.25
- **Total Chunks**: 31
- **Average Chunk Size**: 87.5 words
- **Chunk Size Std Dev**: 67.6

### 3. Semantic-256
- **Average Retrieval Score**: 0.4330
- **Average Coverage**: 1.62
- **Total Chunks**: 24
- **Average Chunk Size**: 60.5 words
- **Chunk Size Std Dev**: 52.4

### 4. Semantic-512
- **Average Retrieval Score**: 0.4049
- **Average Coverage**: 1.62
- **Total Chunks**: 23
- **Average Chunk Size**: 63.2 words
- **Chunk Size Std Dev**: 96.5

### 5. Semantic-1024
- **Average Retrieval Score**: 0.3987
- **Average Coverage**: 1.62
- **Total Chunks**: 22
- **Average Chunk Size**: 66.1 words
- **Chunk Size Std Dev**: 111.6

## Recommendations

**Best Overall Strategy**: Aspect-Based

### Why this strategy works best:
- Excellent for specific aspect queries (crime, schools, transport)
- Reduces noise in retrieval
- Better precision for focused questions

### OpenAI Embedding Model Recommendation:
- **Primary**: `text-embedding-3-large` (3072 dimensions)
  - Best performance for complex real estate queries
  - Superior semantic understanding
- **Alternative**: `text-embedding-3-small` (1536 dimensions)
  - Cost-effective option with good performance
  - Faster processing for large datasets
