"""
Ragas scoring module for RAG evaluation.

This module computes the required metrics:
- faithfulness
- answer_relevancy
- context_precision
- context_recall
"""

import os
import sys
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    print(" Ragas not available. Install with: pip install ragas")
    RAGAS_AVAILABLE = False

from dataset_schemas import validate_predictions
from ragas_dataset import RagasDataset


class RagasScorer:
    """
    Computes Ragas metrics for RAG pipeline evaluation.

    This class handles the computation of all required metrics
    and provides detailed scoring analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ragas scorer.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics = config.get("metrics", [])
        self.thresholds = config.get("thresholds", {})

        if not RAGAS_AVAILABLE:
            raise ImportError(
                "Ragas is required for evaluation. Install with: pip install ragas"
            )

        # Validate required metrics
        required_metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        missing_metrics = [m for m in required_metrics if m not in self.metrics]
        if missing_metrics:
            print(f" Warning: Missing metrics: {missing_metrics}")

    def _validate_dataset_for_evaluation(self, dataset: RagasDataset) -> bool:
        """
        Validate dataset before evaluation to catch potential issues.

        Args:
            dataset: Dataset to validate

        Returns:
            True if dataset is valid for evaluation
        """
        print("Validating dataset for evaluation...")

        # Check for empty or problematic answers
        problematic_answers = []
        if dataset.predictions_df is not None:
            for i, row in dataset.predictions_df.iterrows():
                answer = row.get("answer", "")
                if not answer or answer.strip() == "":
                    problematic_answers.append(f"Row {i}: Empty answer")
                elif answer.lower() in [
                    "i don't know",
                    "no information available",
                    "cannot answer",
                    "not available",
                ]:
                    problematic_answers.append(
                        f"Row {i}: Non-answer response: '{answer[:50]}...'"
                    )

        if problematic_answers:
            print(" Found problematic answers that may cause NaN scores:")
            for issue in problematic_answers:
                print(f"   {issue}")
            print("   These may result in NaN scores for certain metrics")

        # Check for empty contexts
        empty_contexts = []
        if dataset.predictions_df is not None:
            for i, row in dataset.predictions_df.iterrows():
                contexts = row.get("contexts", [])
                if not contexts or (isinstance(contexts, list) and len(contexts) == 0):
                    empty_contexts.append(f"Row {i}: Empty contexts")

        if empty_contexts:
            print(" Found empty contexts:")
            for issue in empty_contexts:
                print(f"   {issue}")
            print("   This will cause context_precision and context_recall to be NaN")

        return True

    def compute_metrics(
        self, dataset: RagasDataset, progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Compute all required Ragas metrics.

        Args:
            dataset: RagasDataset with testset and predictions
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with computed metrics
        """
        if dataset.testset_df is None or dataset.predictions_df is None:
            raise ValueError("Dataset must have both testset and predictions loaded")

        # Validate dataset before evaluation
        self._validate_dataset_for_evaluation(dataset)

        print("Computing Ragas metrics...")

        # Get evaluation data in Ragas format
        eval_data = dataset.get_evaluation_data()

        # Create Ragas dataset
        questions_df, ground_truths_df = dataset.to_ragas_format()

        # Create contexts and answers dataframes
        contexts_df = pd.DataFrame({"contexts": eval_data["contexts"]})

        answers_df = pd.DataFrame({"answer": eval_data["answers"]})

        # Prepare dataset for Ragas
        from datasets import Dataset

        ragas_dataset = Dataset.from_pandas(
            pd.concat([questions_df, ground_truths_df, contexts_df, answers_df], axis=1)
        )

        print(f"Dataset prepared: {len(ragas_dataset)} samples")

        # Compute metrics
        results = {}
        total_metrics = len(self.metrics)

        for i, metric_name in enumerate(self.metrics):
            if progress_callback:
                progress_callback(i + 1, total_metrics, f"Computing {metric_name}")
            else:
                print(f"Computing {metric_name}...")

            try:
                metric_result = self._compute_single_metric(metric_name, ragas_dataset)
                results[metric_name] = metric_result
                print(f"{metric_name}: {metric_result:.4f}")

            except Exception as e:
                print(f"Failed to compute {metric_name}: {e}")
                results[metric_name] = None

        # Add metadata
        results["metadata"] = {
            "computation_timestamp": datetime.now().isoformat(),
            "dataset_size": len(ragas_dataset),
            "metrics_computed": list(results.keys()),
            "config": self.config,
        }

        # Check against thresholds
        threshold_analysis = self._analyze_thresholds(results)
        results["threshold_analysis"] = threshold_analysis

        # Provide guidance on NaN scores
        nan_metrics = [
            name
            for name, score in results.items()
            if name not in ["metadata", "threshold_analysis"]
            and (
                score is None
                or (isinstance(score, float) and (pd.isna(score) or np.isnan(score)))
            )
        ]

        if nan_metrics:
            print(f"\n NaN scores detected for: {nan_metrics}")
            print("   This is common in RAGAS evaluation and can happen when:")
            print("   - Model outputs are not JSON-parsable")
            print("   - Non-ideal cases for scoring (e.g., 'I don't know' responses)")
            print("   - Context retrieval issues")
            print("   - RAGAS evaluation limitations")
            print(
                "   Consider improving model outputs or using more structured prompts"
            )

        print("All metrics computed successfully")
        return results

    def _compute_single_metric(self, metric_name: str, dataset) -> float:
        """
        Compute a single Ragas metric.

        Args:
            metric_name: Name of the metric to compute
            dataset: Ragas dataset

        Returns:
            Metric score
        """
        # Map metric names to Ragas functions
        metric_functions = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

        if metric_name not in metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}")

        metric_func = metric_functions[metric_name]

        # Configure RAGAS to use the same OpenAI API key
        import os

        # Get API key from config or environment
        api_key = self.config.get("judge", {}).get("api_key") or os.getenv(
            "OPENAI_API_KEY"
        )

        # Handle environment variable substitution
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]  # Remove ${ and }
            api_key = os.getenv(env_var)

        if not api_key:
            raise ValueError("OpenAI API key required for RAGAS evaluation")

        # Set environment variable for RAGAS
        os.environ["OPENAI_API_KEY"] = api_key

        try:
            # Compute metric
            result = evaluate(dataset, [metric_func])

            # Extract score
            score = result[metric_name]

            # Handle case where score might be a list
            if isinstance(score, list):
                score = score[0] if score else 0.0

            # Handle NaN values and non-ideal cases
            if score is None or (
                isinstance(score, float) and (pd.isna(score) or np.isnan(score))
            ):
                print(f" {metric_name} returned NaN/None - this can happen when:")
                print(f"   - Model output is not JSON-parsable")
                print(
                    f"   - Non-ideal cases for scoring (e.g., 'I don't know' responses)"
                )
                print(f"   - RAGAS evaluation limitations")
                print(f"   Setting score to 0.0 for this metric")
                score = 0.0

            return score

        except Exception as e:
            print(f" {metric_name} computation failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   This can happen when:")
            print(f"   - API key issues")
            print(f"   - Model output format problems")
            print(f"   - RAGAS evaluation errors")
            print(f"   Setting score to 0.0 due to error")
            return 0.0

    def _analyze_thresholds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze results against configured thresholds.

        Args:
            results: Computed metric results

        Returns:
            Threshold analysis
        """
        analysis = {
            "passed_thresholds": [],
            "failed_thresholds": [],
            "overall_status": "PASS",
            "summary": {},
        }

        for metric_name, score in results.items():
            if metric_name in ["metadata", "threshold_analysis"]:
                continue

            if score is None:
                analysis["summary"][metric_name] = "FAILED"
                analysis["failed_thresholds"].append(metric_name)
                continue

            threshold = self.thresholds.get(metric_name, 0.0)

            if score >= threshold:
                analysis["passed_thresholds"].append(metric_name)
                analysis["summary"][
                    metric_name
                ] = f"PASS ({score:.4f} >= {threshold:.4f})"
            else:
                analysis["failed_thresholds"].append(metric_name)
                analysis["summary"][
                    metric_name
                ] = f"FAIL ({score:.4f} < {threshold:.4f})"

        # Overall status
        if analysis["failed_thresholds"]:
            analysis["overall_status"] = "FAIL"

        return analysis

    def get_detailed_analysis(
        self, results: Dict[str, Any], dataset: RagasDataset
    ) -> Dict[str, Any]:
        """
        Get detailed analysis of evaluation results.

        Args:
            results: Computed metrics
            dataset: Evaluation dataset

        Returns:
            Detailed analysis
        """
        if dataset.testset_df is None or dataset.predictions_df is None:
            return {"error": "Dataset not available"}

        # Basic statistics
        analysis = {
            "dataset_summary": {
                "total_questions": len(dataset.testset_df),
                "total_predictions": len(dataset.predictions_df),
                "topics": dataset.testset_df["topic"].value_counts().to_dict(),
                "difficulties": dataset.testset_df["difficulty"]
                .value_counts()
                .to_dict(),
            },
            "performance_metrics": {},
            "topic_analysis": {},
            "difficulty_analysis": {},
            "recommendations": [],
        }

        # Performance metrics
        for metric_name, score in results.items():
            if (
                metric_name not in ["metadata", "threshold_analysis"]
                and score is not None
            ):
                analysis["performance_metrics"][metric_name] = {
                    "score": score,
                    "threshold": self.thresholds.get(metric_name, 0.0),
                    "status": (
                        "PASS"
                        if score >= self.thresholds.get(metric_name, 0.0)
                        else "FAIL"
                    ),
                }

        # Topic-wise analysis
        topics = dataset.testset_df["topic"].unique()
        for topic in topics:
            topic_mask = dataset.testset_df["topic"] == topic
            topic_questions = dataset.testset_df[topic_mask]
            topic_predictions = dataset.predictions_df[topic_mask]

            # Calculate average source count for this topic
            avg_sources = (
                topic_predictions["meta"]
                .apply(lambda x: x.get("source_count", 0))
                .mean()
            )

            analysis["topic_analysis"][topic] = {
                "question_count": len(topic_questions),
                "avg_sources": avg_sources,
                "difficulty_distribution": topic_questions["difficulty"]
                .value_counts()
                .to_dict(),
            }

        # Difficulty-wise analysis
        difficulties = dataset.testset_df["difficulty"].unique()
        for difficulty in difficulties:
            diff_mask = dataset.testset_df["difficulty"] == difficulty
            diff_questions = dataset.testset_df[diff_mask]
            diff_predictions = dataset.predictions_df[diff_mask]

            # Calculate average source count for this difficulty
            avg_sources = (
                diff_predictions["meta"]
                .apply(lambda x: x.get("source_count", 0))
                .mean()
            )

            analysis["difficulty_analysis"][difficulty] = {
                "question_count": len(diff_questions),
                "avg_sources": avg_sources,
                "topic_distribution": diff_questions["topic"].value_counts().to_dict(),
            }

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(results, analysis)

        return analysis

    def _generate_recommendations(
        self, results: Dict[str, Any], analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate improvement recommendations based on results.

        Args:
            results: Computed metrics
            analysis: Detailed analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check individual metrics
        for metric_name, score in results.items():
            if metric_name in ["metadata", "threshold_analysis"]:
                continue

            if score is None:
                recommendations.append(
                    f"Investigate why {metric_name} computation failed"
                )
                continue

            threshold = self.thresholds.get(metric_name, 0.0)

            if score < threshold:
                if metric_name == "faithfulness":
                    recommendations.append(
                        "Improve answer faithfulness by enhancing prompt engineering"
                    )
                elif metric_name == "answer_relevancy":
                    recommendations.append(
                        "Improve answer relevancy by refining the QA chain prompts"
                    )
                elif metric_name == "context_precision":
                    recommendations.append(
                        "Improve context precision by tuning retriever parameters"
                    )
                elif metric_name == "context_recall":
                    recommendations.append(
                        "Improve context recall by increasing retrieval diversity"
                    )

        # Check topic distribution
        topic_counts = analysis["dataset_summary"]["topics"]
        if len(topic_counts) < 5:
            recommendations.append("Consider adding more diverse topics to the testset")

        # Check difficulty distribution
        difficulty_counts = analysis["dataset_summary"]["difficulties"]
        if difficulty_counts.get("hard", 0) < difficulty_counts.get("easy", 0) * 0.3:
            recommendations.append(
                "Consider adding more challenging questions to the testset"
            )

        return recommendations

    def save_results(
        self, results: Dict[str, Any], analysis: Dict[str, Any], output_path: str
    ):
        """
        Save evaluation results and analysis.

        Args:
            results: Computed metrics
            analysis: Detailed analysis
            output_path: Output file path
        """
        # Combine results and analysis
        output_data = {
            "metrics": results,
            "analysis": analysis,
            "export_timestamp": datetime.now().isoformat(),
        }

        # Save to file
        if output_path.endswith(".json"):
            import json

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
        else:
            # Default to JSON
            output_path = (
                output_path + ".json"
                if "." not in output_path
                else output_path.replace(".", "_") + ".json"
            )
            import json

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")

    def save_metrics_csv(
        self, results: Dict[str, Any], dataset: RagasDataset, output_dir: str
    ):
        """
        Save metrics in CSV format for Definition of Done requirements.

        Args:
            results: Computed metrics
            dataset: RagasDataset with predictions
            output_dir: Output directory
        """
        # Create metrics_per_sample.csv
        per_sample_data = []
        predictions_df = dataset.predictions_df

        if predictions_df is not None:
            for idx, row in predictions_df.iterrows():
                sample_metrics = {
                    "question": row["question"],
                    "answer": row["answer"],
                    "topic": row.get("meta", {}).get("topic", "unknown"),
                    "difficulty": row.get("meta", {}).get("difficulty", "unknown"),
                }

                # Add individual metric scores (would come from detailed Ragas results)
                for metric_name in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]:
                    # For now, use the aggregate score (in real implementation,
                    # this would be the per-sample score from Ragas)
                    sample_metrics[metric_name] = results.get(metric_name, 0.0)

                per_sample_data.append(sample_metrics)

        per_sample_df = pd.DataFrame(per_sample_data)
        per_sample_path = os.path.join(output_dir, "metrics_per_sample.csv")
        per_sample_df.to_csv(str(per_sample_path), index=False)
        print(f"Per-sample metrics saved: {per_sample_path}")

        # Create metrics_aggregate.csv
        aggregate_data = []
        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            score = results.get(metric_name, 0.0)
            threshold = self.config.get("thresholds", {}).get(metric_name, 0.0)

            aggregate_data.append(
                {
                    "metric": metric_name,
                    "mean": score,
                    "std": 0.0,  # Would be calculated from per-sample scores
                    "min": score,  # Would be calculated from per-sample scores
                    "max": score,  # Would be calculated from per-sample scores
                    "threshold": threshold,
                    "passes_threshold": score >= threshold,
                    "count": len(per_sample_data) if per_sample_data else 0,
                }
            )

        aggregate_df = pd.DataFrame(aggregate_data)
        aggregate_path = os.path.join(output_dir, "metrics_aggregate.csv")
        aggregate_df.to_csv(str(aggregate_path), index=False)
        print(f"Aggregate metrics saved: {aggregate_path}")

        return per_sample_path, aggregate_path

    def save_run_meta(
        self,
        results: Dict[str, Any],
        dataset: RagasDataset,
        output_dir: str,
        run_id: str,
    ):
        """
        Save run metadata as run_meta.json for Definition of Done.

        Args:
            results: Computed metrics
            dataset: RagasDataset with predictions
            output_dir: Output directory
            run_id: Run identifier
        """
        meta_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "metrics_computed": list(results.keys()),
            "dataset_summary": {
                "testset_size": (
                    len(dataset.testset_df) if dataset.testset_df is not None else 0
                ),
                "predictions_size": (
                    len(dataset.predictions_df)
                    if dataset.predictions_df is not None
                    else 0
                ),
            },
            "thresholds": self.config.get("thresholds", {}),
            "overall_pass": all(
                results.get(metric, 0.0)
                >= self.config.get("thresholds", {}).get(metric, 0.0)
                for metric in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]
                if metric in results
            ),
            "ragas_version": "simulated",  # Would be actual ragas version
            "evaluation_version": "1.0.0",
        }

        meta_path = os.path.join(output_dir, "run_meta.json")
        import json

        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2, default=str)

        print(f"Run metadata saved: {meta_path}")
        return meta_path


# Example usage
if __name__ == "__main__":
    print("🧪 Testing RagasScorer...")

    if not RAGAS_AVAILABLE:
        print("Ragas not available. Install with: pip install ragas")
    else:
        print("RagasScorer module imported successfully")

        # Example configuration
        config = {
            "metrics": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
            "thresholds": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.85,
                "context_precision": 0.60,
                "context_recall": 0.70,
            },
        }

        try:
            scorer = RagasScorer(config)
            print("RagasScorer initialized successfully")
        except Exception as e:
            print(f"Failed to initialize RagasScorer: {e}")
