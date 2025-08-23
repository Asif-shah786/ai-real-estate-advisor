# RAG Evaluation Module

This module provides comprehensive evaluation capabilities for RAG pipelines, generating all required outputs according to the Definition of Done.

## Quick Start

### Generate Testset Only
```bash
cd eval
python3 run_evaluation.py testset-only
```

### Run Complete Evaluation
```bash
cd eval
python3 run_evaluation.py
```

## Required Outputs (Definition of Done)

The evaluation system generates all required files in `outputs/{run_id}/`:

1. **testset.parquet** - Generated test questions with topics and difficulty
2. **predictions.parquet** - Mock predictions (ready for real RAG pipeline)
3. **metrics_per_sample.csv** - Per-question metric scores
4. **metrics_aggregate.csv** - Aggregated metrics with thresholds
5. **report.html** - Comprehensive HTML report with topic slices and worst examples
6. **run_meta.json** - Run metadata and configuration

## Four Core Metrics

- **Faithfulness** - How well answers stick to retrieved context
- **Answer Relevancy** - How relevant answers are to questions
- **Context Precision** - How precise retrieved contexts are
- **Context Recall** - How complete the retrieved contexts are

## 🔧 Configuration

Edit `configs.yaml` to customize:
- Number of test questions
- Topic distribution
- Difficulty distribution
- Metric thresholds

## Next Steps for Production

1. Install dependencies: `pip install ragas langchain`
2. Set `OPENAI_API_KEY` environment variable
3. Replace mock data with real RAG pipeline execution
4. System is ready for evaluation testing!

## File Structure

```
eval/
├── __init__.py              # Module initialization
├── configs.yaml            # Configuration file
├── dataset_schemas.py      # Data validation schemas
├── ragas_dataset.py        # Dataset handling
├── testset_gen.py          # Testset generation
├── score.py                # Metrics computation
├── reporting.py            # Report generation
├── run_evaluation.py       # Main evaluation runner
├── outputs/                # Generated outputs
└── README.md               # This file
```
