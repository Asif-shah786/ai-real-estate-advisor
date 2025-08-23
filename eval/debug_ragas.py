#!/usr/bin/env python3
"""
Debug script to identify RAGAS evaluation issues.
"""

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
import traceback


def test_ragas_evaluation():
    """Test RAGAS evaluation step by step."""

    print("Debugging RAGAS evaluation...")

    # Load predictions
    try:
        df = pd.read_parquet("outputs/ragas_20250823_220553/predictions.parquet")
        print(f"Loaded predictions: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Failed to load predictions: {e}")
        return

    # Check data quality
    print("\nData Quality Check:")
    for col in ["question", "answer", "contexts"]:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            print(f"   {col}: {null_count} null values")
            # Show sample data
            sample = df[col].iloc[0] if len(df) > 0 else None
            print(f"   {col} sample: {str(sample)[:100]}...")

    # Check for problematic answers
    print("\nChecking for problematic answers:")
    for i, row in df.iterrows():
        answer = row.get("answer", "")
        if not answer or answer.strip() == "":
            print(f"   Row {i}: Empty answer")
        elif answer.lower() in [
            "i don't know",
            "no information available",
            "cannot answer",
        ]:
            print(f"   Row {i}: Non-answer response: '{answer[:50]}...'")

    # Create RAGAS dataset
    try:
        print("\n Creating RAGAS dataset...")
        ragas_dataset = Dataset.from_pandas(df)
        print(f"RAGAS dataset created: {len(ragas_dataset)} samples")
        print(f"Dataset columns: {ragas_dataset.column_names}")
    except Exception as e:
        print(f"Failed to create RAGAS dataset: {e}")
        return

    # Test each metric individually
    metrics_to_test = [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
    ]

    print("\nTesting individual metrics:")
    for metric_name, metric_func in metrics_to_test:
        try:
            print(f"\nTesting {metric_name}...")
            result = evaluate(ragas_dataset, [metric_func])
            score = result[metric_name]

            if score is None:
                print(f"   {metric_name}: None result")
            elif isinstance(score, float) and pd.isna(score):
                print(f"   {metric_name}: NaN result")
            else:
                print(f"   {metric_name}: {score}")

        except Exception as e:
            print(f"   {metric_name} failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            traceback.print_exc()


if __name__ == "__main__":
    test_ragas_evaluation()
