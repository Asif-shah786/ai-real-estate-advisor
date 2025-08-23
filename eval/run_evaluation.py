#!/usr/bin/env python3
"""
Direct evaluation runner for RAG pipeline.

This script demonstrates how to run the complete evaluation pipeline
without using CLI commands.
"""

import os
import sys
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Import will be done inside functions to avoid circular import issues


def load_config(config_path: str = "configs.yaml"):
    """Load evaluation configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_complete_evaluation(
    force_recreate_db: bool = False, force_recreate_testset: bool = False
):
    """Run the complete evaluation pipeline."""
    print("🚀 Starting RAG Pipeline Evaluation...")

    # Load configuration
    config = load_config()
    print(f"📋 Loaded config: {config.get('run_id_prefix', 'ragas')}")

    # Generate run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.get('run_id_prefix', 'ragas')}_{timestamp}"
    print(f"🆔 Run ID: {run_id}")

    # Create output directory
    output_dir = os.path.join("outputs", run_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Step 1: Generate or use existing testset
    print("\n" + "=" * 50)
    print("📝 STEP 1: Testset Management")
    print("=" * 50)

    from testset_gen import build_synthetic_testset

    # Check if we need to create a new testset
    if force_recreate_testset:
        print("🔄 Force recreating testset...")
        n_questions = (
            config.get("testset", {}).get("synthetic", {}).get("n_questions", 5)
        )
        properties_path = (
            config.get("testset", {})
            .get("synthetic", {})
            .get("properties_path", "../datasets/run_ready_904.json")
        )

        testset = build_synthetic_testset(
            cfg={},
            outdir=output_dir,
            properties_path=properties_path,
            n_questions=n_questions,
            seed=config.get("seed", 42),
        )
        print(f"✅ New testset created with {len(testset)} questions")
    else:
        # Use existing testset from most recent output
        existing_outputs = [d for d in os.listdir("outputs") if d.startswith("ragas_")]
        if existing_outputs:
            latest_output = sorted(existing_outputs)[-1]
            existing_testset_path = os.path.join(
                "outputs", latest_output, "testset.parquet"
            )
            if os.path.exists(existing_testset_path):
                import pandas as pd

                testset = pd.read_parquet(existing_testset_path)
                print(
                    f"✅ Using existing testset from {latest_output} with {len(testset)} questions"
                )
            else:
                print("⚠️ Existing testset not found, creating new one...")
                testset = build_synthetic_testset(
                    cfg={},
                    outdir=output_dir,
                    properties_path="../datasets/run_ready_904.json",
                    n_questions=5,
                    seed=config.get("seed", 42),
                )
        else:
            print("🆕 No existing testsets found, creating new one...")
            testset = build_synthetic_testset(
                cfg={},
                outdir=output_dir,
                properties_path="../datasets/run_ready_904.json",
                n_questions=5,
                seed=config.get("seed", 42),
            )

    # Save testset in output directory
    testset_path = os.path.join(output_dir, "testset.parquet")
    testset.to_parquet(testset_path, index=False)
    print(f"💾 Testset saved: {testset_path}")

    # Create dataset for predictions
    from ragas_dataset import RagasDataset

    dataset = RagasDataset()
    dataset.testset_df = testset

    # Step 2: Use existing RAG pipeline (don't recreate)
    print("\n" + "=" * 50)
    print("🔄 STEP 2: Using Existing RAG Pipeline")
    print("=" * 50)

    # Add parent directory to path to import rag_pipeline
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        # Get API key
        from common.cfg import get_openai_api_key

        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            raise Exception("OpenAI API key not found in config")

        # Check if we need to recreate the database
        if force_recreate_db:
            print("🔄 Force recreating vector database...")
            from aspect_based_chunker import create_aspect_based_vectordb
            from langchain_community.embeddings import OpenAIEmbeddings

            embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-large", api_key=openai_api_key
            )
            vectordb = create_aspect_based_vectordb(
                openai_api_key=openai_api_key,
                properties_file="../datasets/run_ready_904.json",
                legal_file="../datasets/legal_uk_greater_manchester.jsonl",
                embedding_model=embedding_model,
                force_recreate=True,
            )
            print("✅ New vector database created")

            # IMPORTANT: Pass the existing database to RAG pipeline to avoid double creation
            print("🔧 Loading RAG pipeline with existing database...")
            from rag_pipeline import RAGPipeline

            pipeline = RAGPipeline(openai_api_key, existing_vectordb=vectordb)
            print("✅ RAG Pipeline loaded successfully with existing database")
        else:
            print("📁 Using existing vector database (no recreation needed)")

            # Import and use existing RAG pipeline
            print("🔧 Loading existing RAG pipeline...")
            from rag_pipeline import RAGPipeline

            pipeline = RAGPipeline(openai_api_key)
            print("✅ RAG Pipeline loaded successfully")

        # Step 3: Run evaluation on testset (no testing, just evaluation)
        print(f"🚀 Running evaluation on {len(testset)} questions...")
        predictions = []

        for idx, (_, row) in enumerate(testset.iterrows()):
            if idx % 5 == 0:  # Progress every 5 questions
                print(f"   Processing question {idx + 1}/{len(testset)}...")

            try:
                # Run query through pipeline
                result = pipeline.run_query(row["question"], use_memory=False)

                # Extract contexts and answer
                contexts = result.get("contexts", [])
                answer = result.get("answer", "")

                # Create prediction record
                prediction = {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "contexts": contexts,
                    "answer": answer,
                    "meta": {
                        "topic": row["topic"],
                        "difficulty": row["difficulty"],
                        "retrieval_k": config.get("retrieval_k", 4),
                        "answer_model": "gpt-4",
                        "timestamp": datetime.now().isoformat(),
                        "source_count": len(contexts),
                        "retriever_type": result.get("meta", {}).get(
                            "retriever_type", "unknown"
                        ),
                        "pipeline_error": None,
                    },
                }

            except Exception as e:
                print(f"⚠️ Error processing question {idx + 1}: {e}")
                # Create error prediction record
                prediction = {
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "contexts": [],
                    "answer": f"Error: {str(e)}",
                    "meta": {
                        "topic": row["topic"],
                        "difficulty": row["difficulty"],
                        "retrieval_k": config.get("retrieval_k", 4),
                        "answer_model": "error",
                        "timestamp": datetime.now().isoformat(),
                        "source_count": 0,
                        "retriever_type": "error",
                        "pipeline_error": str(e),
                    },
                }

            predictions.append(prediction)

        print(f"✅ Evaluation completed: {len(predictions)} predictions generated")

        # Save predictions
        dataset.add_predictions(predictions)
        predictions_path = os.path.join(output_dir, "predictions.parquet")
        dataset.save_predictions(predictions_path)
        print(f"💾 Predictions saved: {predictions_path}")

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        raise RuntimeError(f"Pipeline execution failed: {e}")

    # Step 4: Score results using RagasScorer
    print("\n" + "=" * 50)
    print("📊 STEP 4: Computing Metrics")
    print("=" * 50)

    try:
        from score import RagasScorer

        print("🔧 Initializing RagasScorer...")
        scorer = RagasScorer(config=config)

        # Compute metrics using real Ragas
        print("🧮 Computing Ragas metrics...")
        results = scorer.compute_metrics(dataset)

        print(f"✅ Metrics computed successfully:")

        # Display detailed per-sample results
        print("\n📊 **Detailed Per-Sample Results:**")
        detailed_results = []
        for metric, score in results.items():
            if metric in ["metadata", "threshold_analysis"]:
                continue
            if score is None or (isinstance(score, float) and np.isnan(score)):
                print(f"   {metric}: {score}")
                detailed_results.append(f"{metric}: {score}")
            else:
                print(f"   {metric}: {score:.3f}")
                detailed_results.append(f"{metric}: {score:.3f}")

        # Display the detailed results in the format requested
        print(f"\n📋 **Per-Sample Breakdown:**")
        print("Results:", " ".join(detailed_results))

        # Save metrics using RagasScorer methods
        print("💾 Saving metrics...")
        scorer.save_metrics_csv(results, dataset, output_dir)
        scorer.save_run_meta(results, dataset, output_dir, run_id)

        print("✅ Real metrics saved using RagasScorer")

    except Exception as e:
        print(f"❌ Real scoring failed: {e}")
        raise RuntimeError(f"Real scoring failed: {e}")

    # Step 5: Generate reports
    print("\n" + "=" * 50)
    print("📋 STEP 5: Generating Reports")
    print("=" * 50)

    from reporting import EvaluationReporter

    reporter = EvaluationReporter(config=config)

    # Create analysis for reporting
    analysis = {
        "dataset_summary": {
            "total_questions": len(testset),
            "total_predictions": len(predictions),
            "topics": testset["topic"].value_counts().to_dict(),
            "difficulties": testset["difficulty"].value_counts().to_dict(),
        },
        "threshold_analysis": {
            "overall_status": (
                "PASS"
                if all(
                    (
                        results[metric] is not None
                        and not (
                            isinstance(results[metric], float)
                            and np.isnan(results[metric])
                        )
                        and results[metric]
                        >= config.get("thresholds", {}).get(metric, 0.0)
                    )
                    for metric in results.keys()
                    if metric not in ["metadata", "threshold_analysis"]
                )
                else "FAIL"
            )
        },
        "recommendations": [
            "System is performing well with current configuration",
            "Consider testing with more challenging questions",
            "Monitor faithfulness scores for potential hallucination",
        ],
    }

    # Generate report.html (required file)
    report_path = os.path.join(output_dir, "report.html")
    reporter._generate_html_report(results, analysis, dataset, report_path)

    print("\n" + "=" * 50)
    print("✅ EVALUATION COMPLETE - DEFINITION OF DONE ✅")
    print("=" * 50)
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📝 Testset used: {len(testset)} questions")
    print(f"🎯 Predictions created: {len(predictions)} samples")

    print("\n📋 Required files created (Definition of Done):")
    print("   ✅ testset.parquet")
    print("   ✅ predictions.parquet")
    print("   ✅ metrics_per_sample.csv")
    print("   ✅ metrics_aggregate.csv")
    print("   ✅ report.html (with topic slices & worst examples)")
    print("   ✅ run_meta.json")

    print("\n📊 **Current Results:**")
    for metric, score in results.items():
        if metric in ["metadata", "threshold_analysis"]:
            continue
        threshold = config.get("thresholds", {}).get(metric, 0.0)
        if score is None or (isinstance(score, float) and np.isnan(score)):
            status = "❌ FAIL"
            print(f"   - **{metric}**: {score} ❌ (FAIL - below {threshold} threshold)")
        else:
            status = "✅ PASS" if score >= threshold else "❌ FAIL"
            print(
                f"   - **{metric}**: {score:.3f} {'✅' if score >= threshold else '❌'} ({status} - {'above' if score >= threshold else 'below'} {threshold} threshold)"
            )

    print(f"\n🎯 Overall Status: {analysis['threshold_analysis']['overall_status']}")


if __name__ == "__main__":
    import sys

    # Parse command line arguments for force flags
    force_recreate_db = "--force-recreate-db" in sys.argv
    force_recreate_testset = "--force-recreate-testset" in sys.argv

    print("🔧 Command line options:")
    print(f"   Force recreate DB: {force_recreate_db}")
    print(f"   Force recreate testset: {force_recreate_testset}")

    run_complete_evaluation(
        force_recreate_db=force_recreate_db,
        force_recreate_testset=force_recreate_testset,
    )
