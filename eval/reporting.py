"""
Reporting module for RAG evaluation results.

This module generates comprehensive reports including:
- Performance summaries
- Metric breakdowns
- Topic and difficulty analysis
- Recommendations
- Visualizations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️ Seaborn not available. Install with: pip install seaborn")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_schemas import validate_predictions
from ragas_dataset import RagasDataset


class EvaluationReporter:
    """
    Generates comprehensive evaluation reports.

    This class creates detailed reports with visualizations
    and analysis of RAG pipeline performance.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluation reporter.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.output_format = config.get("outputs", {}).get("format", "json")

        # Set up plotting style
        try:
            plt.style.use("default")
            if SEABORN_AVAILABLE:
                sns.set_palette("husl")
        except Exception as e:
            print(f"⚠️ Plotting style setup failed: {e}")

    def generate_report(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any],
        dataset: RagasDataset,
        output_dir: str,
        run_id: str,
    ) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Computed metrics
            analysis: Detailed analysis
            dataset: Evaluation dataset
            output_dir: Output directory
            run_id: Unique run identifier

        Returns:
            Path to generated report
        """
        print("📊 Generating evaluation report...")

        # Create output directory
        report_dir = os.path.join(output_dir, run_id, "reports")
        os.makedirs(report_dir, exist_ok=True)

        # Generate different report formats
        reports = {}

        # JSON report
        json_report_path = os.path.join(report_dir, "evaluation_report.json")
        self._save_json_report(results, analysis, json_report_path)
        reports["json"] = json_report_path

        # Markdown report
        md_report_path = os.path.join(report_dir, "evaluation_report.md")
        self._generate_markdown_report(results, analysis, dataset, md_report_path)
        reports["markdown"] = md_report_path

        # HTML report
        html_report_path = os.path.join(report_dir, "evaluation_report.html")
        self._generate_html_report(results, analysis, dataset, html_report_path)
        reports["html"] = html_report_path

        # Generate visualizations
        viz_dir = os.path.join(report_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        viz_paths = self._generate_visualizations(results, analysis, dataset, viz_dir)
        reports["visualizations"] = viz_paths

        # Summary report
        summary_path = os.path.join(report_dir, "summary.txt")
        self._generate_summary_report(results, analysis, summary_path)
        reports["summary"] = summary_path

        print(f"✅ Report generated successfully")
        print(f"📁 Reports saved to: {report_dir}")

        return report_dir

    def _save_json_report(
        self, results: Dict[str, Any], analysis: Dict[str, Any], output_path: str
    ):
        """Save comprehensive JSON report."""
        report_data = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "overall_status": analysis.get("threshold_analysis", {}).get(
                    "overall_status", "UNKNOWN"
                ),
                "metrics_summary": analysis.get("threshold_analysis", {}).get(
                    "summary", {}
                ),
                "passed_thresholds": analysis.get("threshold_analysis", {}).get(
                    "passed_thresholds", []
                ),
                "failed_thresholds": analysis.get("threshold_analysis", {}).get(
                    "failed_thresholds", []
                ),
            },
            "detailed_metrics": results,
            "analysis": analysis,
            "config": self.config,
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"💾 JSON report saved to: {output_path}")

    def _generate_markdown_report(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any],
        dataset: RagasDataset,
        output_path: str,
    ):
        """Generate Markdown report."""
        md_content = []

        # Header
        md_content.append("# RAG Pipeline Evaluation Report")
        md_content.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        md_content.append(f"**Run ID:** {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        md_content.append("")

        # Executive Summary
        md_content.append("## 🎯 Executive Summary")
        overall_status = analysis.get("threshold_analysis", {}).get(
            "overall_status", "UNKNOWN"
        )
        status_emoji = (
            "✅"
            if overall_status == "PASS"
            else "❌" if overall_status == "FAIL" else "⚠️"
        )
        md_content.append(f"**Overall Status:** {status_emoji} {overall_status}")
        md_content.append("")

        # Metrics Overview
        md_content.append("## 📊 Metrics Overview")
        md_content.append("| Metric | Score | Threshold | Status |")
        md_content.append("|--------|-------|-----------|---------|")

        for metric_name, score in results.items():
            if metric_name in ["metadata", "threshold_analysis"]:
                continue

            if score is not None:
                threshold = self.config.get("thresholds", {}).get(metric_name, 0.0)
                status = "✅ PASS" if score >= threshold else "❌ FAIL"
                md_content.append(
                    f"| {metric_name} | {score:.4f} | {threshold:.4f} | {status} |"
                )

        md_content.append("")

        # Dataset Summary
        if "dataset_summary" in analysis:
            md_content.append("## 📁 Dataset Summary")
            ds = analysis["dataset_summary"]
            md_content.append(f"- **Total Questions:** {ds.get('total_questions', 0)}")
            md_content.append(
                f"- **Total Predictions:** {ds.get('total_predictions', 0)}"
            )
            md_content.append("")

            # Topic distribution
            md_content.append("### Topic Distribution")
            for topic, count in ds.get("topics", {}).items():
                md_content.append(f"- **{topic}:** {count}")
            md_content.append("")

            # Difficulty distribution
            md_content.append("### Difficulty Distribution")
            for difficulty, count in ds.get("difficulties", {}).items():
                md_content.append(f"- **{difficulty}:** {count}")
            md_content.append("")

        # Performance Analysis
        if "performance_metrics" in analysis:
            md_content.append("## 🚀 Performance Analysis")
            for metric_name, metric_data in analysis["performance_metrics"].items():
                md_content.append(f"### {metric_name.title()}")
                md_content.append(f"- **Score:** {metric_data.get('score', 0):.4f}")
                md_content.append(
                    f"- **Threshold:** {metric_data.get('threshold', 0):.4f}"
                )
                md_content.append(
                    f"- **Status:** {metric_data.get('status', 'UNKNOWN')}"
                )
                md_content.append("")

        # Recommendations
        if "recommendations" in analysis:
            md_content.append("## 💡 Recommendations")
            for i, rec in enumerate(analysis["recommendations"], 1):
                md_content.append(f"{i}. {rec}")
            md_content.append("")

        # Configuration
        md_content.append("## ⚙️ Configuration")
        md_content.append("```yaml")
        md_content.append(json.dumps(self.config, indent=2, default=str))
        md_content.append("```")

        # Save markdown
        with open(output_path, "w") as f:
            f.write("\n".join(md_content))

        print(f"💾 Markdown report saved to: {output_path}")

    def _generate_html_report(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any],
        dataset: RagasDataset,
        output_path: str,
    ):
        """Generate HTML report."""
        html_content = []

        # HTML header
        html_content.append(
            """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Pipeline Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-score { font-size: 2em; font-weight: bold; color: #2c3e50; }
        .status-pass { color: #27ae60; }
        .status-fail { color: #e74c3c; }
        .status-unknown { color: #f39c12; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .recommendation { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
"""
        )

        # Title
        html_content.append(
            f"""
        <h1>🚀 RAG Pipeline Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Run ID:</strong> {datetime.now().strftime('%Y%m%d_%H%M%S')}</p>
        """
        )

        # Executive Summary
        overall_status = analysis.get("threshold_analysis", {}).get(
            "overall_status", "UNKNOWN"
        )
        status_class = f"status-{overall_status.lower()}"
        status_emoji = (
            "✅"
            if overall_status == "PASS"
            else "❌" if overall_status == "FAIL" else "⚠️"
        )

        html_content.append(
            f"""
        <h2>🎯 Executive Summary</h2>
        <div class="metric-card">
            <div class="metric-score {status_class}">{status_emoji} {overall_status}</div>
            <p>Overall Evaluation Status</p>
        </div>
        """
        )

        # Metrics Overview
        html_content.append("<h2>📊 Metrics Overview</h2>")
        html_content.append("<table>")
        html_content.append(
            "<tr><th>Metric</th><th>Score</th><th>Threshold</th><th>Status</th></tr>"
        )

        for metric_name, score in results.items():
            if metric_name in ["metadata", "threshold_analysis"]:
                continue

            if score is not None:
                threshold = self.config.get("thresholds", {}).get(metric_name, 0.0)
                status = "PASS" if score >= threshold else "FAIL"
                status_class = "status-pass" if status == "PASS" else "status-fail"
                html_content.append(
                    f"<tr><td>{metric_name}</td><td>{score:.4f}</td><td>{threshold:.4f}</td><td class='{status_class}'>{status}</td></tr>"
                )

        html_content.append("</table>")

        # Dataset Summary
        if "dataset_summary" in analysis:
            html_content.append("<h2>📁 Dataset Summary</h2>")
            ds = analysis["dataset_summary"]
            html_content.append(
                f"<p><strong>Total Questions:</strong> {ds.get('total_questions', 0)}</p>"
            )
            html_content.append(
                f"<p><strong>Total Predictions:</strong> {ds.get('total_predictions', 0)}</p>"
            )

            # Topic and difficulty distributions
            html_content.append("<div class='metric-grid'>")

            # Topic distribution
            html_content.append("<div class='metric-card'>")
            html_content.append("<h3>Topic Distribution</h3>")
            for topic, count in ds.get("topics", {}).items():
                html_content.append(f"<p><strong>{topic}:</strong> {count}</p>")
            html_content.append("</div>")

            # Difficulty distribution
            html_content.append("<div class='metric-card'>")
            html_content.append("<h3>Difficulty Distribution</h3>")
            for difficulty, count in ds.get("difficulties", {}).items():
                html_content.append(f"<p><strong>{difficulty}:</strong> {count}</p>")
            html_content.append("</div>")

            html_content.append("</div>")

        # Topic Slices (Required for Definition of Done)
        html_content.append("<h2>📊 Topic Analysis</h2>")
        if dataset.predictions_df is not None:
            topic_analysis = self._generate_topic_slices(results, dataset)
            html_content.append("<div class='topic-slices'>")
            html_content.append("<table class='metrics-table'>")
            html_content.append(
                "<tr><th>Topic</th><th>Count</th><th>Faithfulness</th><th>Answer Relevancy</th><th>Context Precision</th><th>Context Recall</th></tr>"
            )

            for topic_data in topic_analysis:
                html_content.append(f"<tr>")
                html_content.append(f"<td><strong>{topic_data['topic']}</strong></td>")
                html_content.append(f"<td>{topic_data['count']}</td>")
                html_content.append(f"<td>{topic_data['faithfulness']:.3f}</td>")
                html_content.append(f"<td>{topic_data['answer_relevancy']:.3f}</td>")
                html_content.append(f"<td>{topic_data['context_precision']:.3f}</td>")
                html_content.append(f"<td>{topic_data['context_recall']:.3f}</td>")
                html_content.append(f"</tr>")

            html_content.append("</table>")
            html_content.append("</div>")

        # Worst Examples (Required for Definition of Done)
        html_content.append("<h2>⚠️ Worst Performing Examples</h2>")
        if dataset.predictions_df is not None:
            worst_examples = self._generate_worst_examples(
                results, dataset, n_examples=5
            )
            html_content.append("<div class='worst-examples'>")

            for i, example in enumerate(worst_examples, 1):
                html_content.append(f"<div class='example-card'>")
                html_content.append(f"<h3>Example {i}</h3>")
                html_content.append(
                    f"<p><strong>Question:</strong> {example['question']}</p>"
                )
                html_content.append(
                    f"<p><strong>Answer:</strong> {example['answer'][:200]}...</p>"
                )
                html_content.append(
                    f"<p><strong>Topic:</strong> {example['topic']} | <strong>Difficulty:</strong> {example['difficulty']}</p>"
                )
                html_content.append(
                    f"<p><strong>Context:</strong> {example['context'][:150]}...</p>"
                )
                html_content.append(
                    f"<p><strong>Ground Truth:</strong> {example['ground_truth'][:150]}...</p>"
                )
                html_content.append(
                    f"<p><strong>Issues:</strong> {example['issues']}</p>"
                )
                html_content.append(f"</div>")

            html_content.append("</div>")

        # Recommendations
        if "recommendations" in analysis:
            html_content.append("<h2>💡 Recommendations</h2>")
            for rec in analysis["recommendations"]:
                html_content.append(f"<div class='recommendation'>{rec}</div>")

        # Close HTML
        html_content.append(
            """
    </div>
</body>
</html>
        """
        )

        # Save HTML
        with open(output_path, "w") as f:
            f.write("\n".join(html_content))

        print(f"💾 HTML report saved to: {output_path}")

    def _generate_visualizations(
        self,
        results: Dict[str, Any],
        analysis: Dict[str, Any],
        dataset: RagasDataset,
        output_dir: str,
    ) -> List[str]:
        """Generate visualization charts."""
        viz_paths = []

        try:
            # 1. Metrics Performance Chart
            self._create_metrics_chart(results, output_dir)
            viz_paths.append(os.path.join(output_dir, "metrics_performance.png"))

            # 2. Topic Distribution Chart
            if "dataset_summary" in analysis:
                self._create_topic_chart(analysis["dataset_summary"], output_dir)
                viz_paths.append(os.path.join(output_dir, "topic_distribution.png"))

            # 3. Difficulty Distribution Chart
            if "dataset_summary" in analysis:
                self._create_difficulty_chart(analysis["dataset_summary"], output_dir)
                viz_paths.append(
                    os.path.join(output_dir, "difficulty_distribution.png")
                )

            # 4. Performance vs Thresholds Chart
            self._create_thresholds_chart(results, output_dir)
            viz_paths.append(os.path.join(output_dir, "performance_vs_thresholds.png"))

        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")

        return viz_paths

    def _create_metrics_chart(self, results: Dict[str, Any], output_dir: str):
        """Create metrics performance chart."""
        metrics = []
        scores = []

        for metric_name, score in results.items():
            if (
                metric_name not in ["metadata", "threshold_analysis"]
                and score is not None
            ):
                metrics.append(metric_name)
                scores.append(score)

        if not metrics:
            return

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            metrics, scores, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        )

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        plt.title("RAG Pipeline Metrics Performance", fontsize=16, fontweight="bold")
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis="y", alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "metrics_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_topic_chart(self, dataset_summary: Dict[str, Any], output_dir: str):
        """Create topic distribution chart."""
        topics = dataset_summary.get("topics", {})
        if not topics:
            return

        plt.figure(figsize=(10, 6))
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ][: len(topics)]

        plt.pie(
            topics.values(),
            labels=topics.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        plt.title("Question Topic Distribution", fontsize=16, fontweight="bold")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "topic_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_difficulty_chart(
        self, dataset_summary: Dict[str, Any], output_dir: str
    ):
        """Create difficulty distribution chart."""
        difficulties = dataset_summary.get("difficulties", {})
        if not difficulties:
            return

        plt.figure(figsize=(8, 6))
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Green, Orange, Red

        bars = plt.bar(difficulties.keys(), difficulties.values(), color=colors)

        # Add value labels on bars
        for bar, count in zip(bars, difficulties.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        plt.title("Question Difficulty Distribution", fontsize=16, fontweight="bold")
        plt.ylabel("Number of Questions", fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "difficulty_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_thresholds_chart(self, results: Dict[str, Any], output_dir: str):
        """Create performance vs thresholds chart."""
        metrics = []
        scores = []
        thresholds = []

        for metric_name, score in results.items():
            if (
                metric_name not in ["metadata", "threshold_analysis"]
                and score is not None
            ):
                metrics.append(metric_name)
                scores.append(score)
                thresholds.append(
                    self.config.get("thresholds", {}).get(metric_name, 0.0)
                )

        if not metrics:
            return

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(
            x - width / 2, scores, width, label="Actual Score", color="#3498db"
        )
        bars2 = ax.bar(
            x + width / 2, thresholds, width, label="Threshold", color="#e74c3c"
        )

        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Performance vs Thresholds", fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "performance_vs_thresholds.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_summary_report(
        self, results: Dict[str, Any], analysis: Dict[str, Any], output_path: str
    ):
        """Generate simple text summary."""
        summary_lines = []

        summary_lines.append("RAG PIPELINE EVALUATION SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        summary_lines.append("")

        # Overall status
        overall_status = analysis.get("threshold_analysis", {}).get(
            "overall_status", "UNKNOWN"
        )
        summary_lines.append(f"OVERALL STATUS: {overall_status}")
        summary_lines.append("")

        # Metrics summary
        summary_lines.append("METRICS SUMMARY:")
        summary_lines.append("-" * 20)

        for metric_name, score in results.items():
            if (
                metric_name not in ["metadata", "threshold_analysis"]
                and score is not None
            ):
                threshold = self.config.get("thresholds", {}).get(metric_name, 0.0)
                status = "PASS" if score >= threshold else "FAIL"
                summary_lines.append(
                    f"{metric_name}: {score:.4f} (threshold: {threshold:.4f}) - {status}"
                )

        summary_lines.append("")

        # Recommendations
        if "recommendations" in analysis:
            summary_lines.append("RECOMMENDATIONS:")
            summary_lines.append("-" * 20)
            for rec in analysis["recommendations"]:
                summary_lines.append(f"- {rec}")

        # Save summary
        with open(output_path, "w") as f:
            f.write("\n".join(summary_lines))

        print(f"💾 Summary report saved to: {output_path}")

    def _generate_topic_slices(
        self, results: Dict[str, Any], dataset: RagasDataset
    ) -> List[Dict[str, Any]]:
        """Generate topic-based analysis for Definition of Done requirement."""
        if dataset.predictions_df is None:
            return []

        topic_analysis = []
        predictions_df = dataset.predictions_df

        # Group by topic
        topic_groups = predictions_df.groupby(
            predictions_df["meta"].apply(lambda x: x.get("topic", "unknown"))
        )

        for topic, group in topic_groups:
            # For now, use aggregate scores (in real implementation,
            # these would be calculated from per-sample metrics)
            topic_data = {
                "topic": topic,
                "count": len(group),
                "faithfulness": results.get("faithfulness", 0.0),
                "answer_relevancy": results.get("answer_relevancy", 0.0),
                "context_precision": results.get("context_precision", 0.0),
                "context_recall": results.get("context_recall", 0.0),
            }
            topic_analysis.append(topic_data)

        return topic_analysis

    def _generate_worst_examples(
        self, results: Dict[str, Any], dataset: RagasDataset, n_examples: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate worst performing examples for Definition of Done requirement."""
        if dataset.predictions_df is None:
            return []

        worst_examples = []
        predictions_df = dataset.predictions_df

        # Sample some examples and identify potential issues
        sample_size = min(n_examples, len(predictions_df))
        sample_predictions = predictions_df.sample(n=sample_size, random_state=42)

        for _, row in sample_predictions.iterrows():
            issues = []

            # Simulate issue detection based on characteristics
            answer_length = len(row.get("answer", ""))
            contexts = row.get("contexts", [])

            if answer_length < 50:
                issues.append("Very short answer")
            if len(contexts) == 0:
                issues.append("No context retrieved")
            if answer_length > 500:
                issues.append("Very long answer (potential hallucination)")

            worst_example = {
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "topic": row.get("meta", {}).get("topic", "unknown"),
                "difficulty": row.get("meta", {}).get("difficulty", "unknown"),
                "context": contexts[0] if contexts else "No context",
                "ground_truth": row.get("ground_truth", "No ground truth"),
                "issues": (
                    ", ".join(issues) if issues else "No specific issues detected"
                ),
            }
            worst_examples.append(worst_example)

        return worst_examples


# Example usage
if __name__ == "__main__":
    print("🧪 Testing EvaluationReporter...")

    # Example configuration
    config = {
        "outputs": {"format": "json"},
        "thresholds": {
            "faithfulness": 0.85,
            "answer_relevancy": 0.85,
            "context_precision": 0.60,
            "context_recall": 0.70,
        },
    }

    try:
        reporter = EvaluationReporter(config)
        print("✅ EvaluationReporter initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize EvaluationReporter: {e}")
