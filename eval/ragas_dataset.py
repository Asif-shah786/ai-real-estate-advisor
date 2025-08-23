"""
Ragas dataset handling for evaluation.

This module handles the conversion of our custom dataset format
to Ragas-compatible datasets and manages the evaluation process.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import json

from dataset_schemas import (
    validate_testset,
    validate_predictions,
    TestsetRow,
    PredictionRow,
)


class RagasDataset:
    """
    Handles Ragas-compatible datasets for evaluation.

    This class manages the conversion between our custom format
    and the format expected by Ragas evaluation functions.
    """

    def __init__(self, testset_path: Optional[str] = None):
        """
        Initialize the Ragas dataset handler.

        Args:
            testset_path: Path to testset file (optional)
        """
        self.testset_path = testset_path
        self.testset_df = None
        self.predictions_df = None

    def load_testset(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load testset from file.

        Args:
            path: Path to testset file

        Returns:
            Loaded and validated testset dataframe
        """
        file_path = path or self.testset_path
        if not file_path:
            raise ValueError("No testset path provided")

        # Determine file type and load
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Validate the testset
        self.testset_df = validate_testset(df)
        print(f"✅ Loaded testset: {len(self.testset_df)} questions")
        return self.testset_df

    def create_synthetic_testset(
        self, n_questions: int = 5, seed: int = 42
    ) -> pd.DataFrame:
        """
        Create a synthetic testset for evaluation.

        Args:
            n_questions: Number of questions to generate
            seed: Random seed for reproducibility

        Returns:
            Synthetic testset dataframe
        """
        np.random.seed(seed)

        # Define question templates by topic
        question_templates = {
            "search": [
                "What properties are available in {location}?",
                "Show me {property_type} properties in {location}",
                "What's the price range for properties in {location}?",
                "Are there any {bedrooms}-bedroom properties in {location}?",
                "What properties are available near {amenity} in {location}?",
            ],
            "amenities": [
                "What amenities are available near {location}?",
                "Are there good schools in {location}?",
                "What's the transport connectivity in {location}?",
                "Are there shopping centers near {location}?",
                "What recreational facilities are available in {location}?",
            ],
            "crime": [
                "What's the crime rate in {location}?",
                "Is {location} a safe area to live?",
                "What are the safety statistics for {location}?",
                "Are there any crime hotspots in {location}?",
                "What's the police presence like in {location}?",
            ],
            "legal": [
                "What legal documents do I need for property purchase in {location}?",
                "What are the property laws in {location}?",
                "What's the tenure situation in {location}?",
                "Are there any legal restrictions in {location}?",
                "What are the property taxes in {location}?",
            ],
            "summary": [
                "Give me a summary of {location}",
                "What's {location} like as a place to live?",
                "Describe the neighborhood of {location}",
                "What are the pros and cons of {location}?",
                "Give me an overview of {location}",
            ],
        }

        # Define topic distribution
        topic_distribution = {
            "search": 0.30,
            "amenities": 0.25,
            "crime": 0.20,
            "legal": 0.15,
            "summary": 0.10,
        }

        # Sample locations and property details
        locations = [
            "Manchester",
            "Salford",
            "Stockport",
            "Trafford",
            "Bolton",
            "Bury",
            "Oldham",
            "Rochdale",
            "Tameside",
            "Wigan",
        ]

        property_types = [
            "detached",
            "semi-detached",
            "terraced",
            "apartment",
            "bungalow",
        ]
        amenities = [
            "schools",
            "hospitals",
            "shopping centers",
            "parks",
            "train stations",
        ]
        bedrooms = [1, 2, 3, 4, 5]

        # Generate questions
        questions = []
        for _ in range(n_questions):
            # Sample topic based on distribution
            topic = np.random.choice(
                list(topic_distribution.keys()), p=list(topic_distribution.values())
            )

            # Sample template and fill with random values
            template = np.random.choice(question_templates[topic])

            # Fill template variables
            question = template.format(
                location=np.random.choice(locations),
                property_type=np.random.choice(property_types),
                bedrooms=np.random.choice(bedrooms),
                amenity=np.random.choice(amenities),
            )

            # Generate difficulty (correlated with topic complexity)
            difficulty_weights = {
                "search": [0.4, 0.4, 0.2],  # Mostly easy/medium
                "amenities": [0.3, 0.5, 0.2],  # Balanced
                "crime": [0.2, 0.4, 0.4],  # More medium/hard
                "legal": [0.1, 0.3, 0.6],  # Mostly hard
                "summary": [0.2, 0.5, 0.3],  # Balanced
            }

            difficulty = np.random.choice(
                ["easy", "medium", "hard"], p=difficulty_weights[topic]
            )

            questions.append(
                {
                    "question": question,
                    "ground_truth": "",  # Empty for synthetic data
                    "topic": topic,
                    "difficulty": difficulty,
                }
            )

        # Create and validate dataframe
        self.testset_df = pd.DataFrame(questions)
        self.testset_df = validate_testset(self.testset_df)

        print(f"✅ Created synthetic testset: {len(self.testset_df)} questions")
        print(
            f"📊 Topic distribution: {self.testset_df['topic'].value_counts().to_dict()}"
        )
        print(
            f"📊 Difficulty distribution: {self.testset_df['difficulty'].value_counts().to_dict()}"
        )

        return self.testset_df

    def add_predictions(self, predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Add predictions to the dataset.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Predictions dataframe
        """
        if self.testset_df is None:
            raise ValueError("No testset loaded. Load testset first.")

        # Convert predictions to dataframe
        predictions_df = pd.DataFrame(predictions)

        # Validate predictions
        self.predictions_df = validate_predictions(predictions_df)

        print(f"✅ Added predictions: {len(self.predictions_df)} rows")
        return self.predictions_df

    def to_ragas_format(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert to Ragas-compatible format.

        Returns:
            Tuple of (questions_df, ground_truths_df) for Ragas
        """
        if self.testset_df is None:
            raise ValueError("No testset loaded")

        # Create questions dataframe
        questions_df = pd.DataFrame({"question": self.testset_df["question"]})

        # Create ground truths dataframe (can be empty for synthetic data)
        ground_truths_df = pd.DataFrame(
            {"ground_truth": self.testset_df["ground_truth"]}
        )

        return questions_df, ground_truths_df

    def get_evaluation_data(self) -> Dict[str, Any]:
        """
        Get data ready for Ragas evaluation.

        Returns:
            Dictionary with evaluation data
        """
        if self.testset_df is None:
            raise ValueError("No testset loaded")

        if self.predictions_df is None:
            raise ValueError("No predictions added")

        # Ensure alignment between testset and predictions
        if len(self.testset_df) != len(self.predictions_df):
            raise ValueError("Testset and predictions have different lengths")

        # Create evaluation data
        eval_data = {
            "questions": self.testset_df["question"].tolist(),
            "ground_truths": self.testset_df["ground_truth"].tolist(),
            "contexts": self.predictions_df["contexts"].tolist(),
            "answers": self.predictions_df["answer"].tolist(),
            "topics": self.testset_df["topic"].tolist(),
            "difficulties": self.testset_df["difficulty"].tolist(),
            "metadata": self.predictions_df["meta"].tolist(),
        }

        return eval_data

    def save_testset(self, path: str, format: str = "parquet"):
        """Save testset to file."""
        if self.testset_df is None:
            raise ValueError("No testset to save")

        if format == "parquet":
            self.testset_df.to_parquet(path, index=False)
        elif format == "csv":
            self.testset_df.to_csv(path, index=False)
        elif format == "json":
            self.testset_df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✅ Saved testset to: {path}")

    def save_predictions(self, path: str, format: str = "parquet"):
        """Save predictions to file."""
        if self.predictions_df is None:
            raise ValueError("No predictions to save")

        if format == "parquet":
            self.predictions_df.to_parquet(path, index=False)
        elif format == "csv":
            self.predictions_df.to_csv(path, index=False)
        elif format == "json":
            self.predictions_df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"✅ Saved predictions to: {path}")


# Example usage
if __name__ == "__main__":
    print("🧪 Testing RagasDataset...")

    # Create dataset handler
    dataset = RagasDataset()

    # Create synthetic testset
    testset = dataset.create_synthetic_testset(n_questions=5, seed=42)

    # Save testset
    dataset.save_testset("outputs/test_synthetic.parquet")

    print("✅ RagasDataset tests completed")
