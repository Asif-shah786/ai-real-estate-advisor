"""
Dataset schemas for Ragas evaluation.

This module defines strict schemas for testset and prediction data
using Pydantic for validation and type safety.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np

# Allowed topics for testset questions
ALLOWED_TOPICS = [
    "search",
    "amenities",
    "crime",
    "legal",
    "summary",
    "other",
    "property-facts",
]


class TestsetRow(BaseModel):
    """Schema for a single testset row."""

    question: str = Field(..., min_length=1, description="The question to evaluate")
    ground_truth: str = Field(
        default="", description="Ground truth answer (can be empty)"
    )
    topic: str = Field(..., description="Topic category of the question")
    difficulty: str = Field(..., description="Difficulty level: easy, medium, hard")

    @validator("difficulty")
    def validate_difficulty(cls, v):
        allowed = ["easy", "medium", "hard"]
        if v.lower() not in allowed:
            raise ValueError(f"Difficulty must be one of {allowed}")
        return v.lower()

    @validator("topic")
    def validate_topic(cls, v):
        if v.lower() not in ALLOWED_TOPICS:
            raise ValueError(f"Topic must be one of {ALLOWED_TOPICS}")
        return v.lower()


class PredictionRow(BaseModel):
    """Schema for a single prediction row."""

    question: str = Field(..., min_length=1, description="The question that was asked")
    ground_truth: str = Field(default="", description="Ground truth answer")
    contexts: List[str] = Field(..., description="Retrieved contexts from RAG pipeline")
    answer: str = Field(..., description="Generated answer from RAG pipeline")
    meta: Dict[str, Any] = Field(..., description="Additional metadata")

    @validator("contexts")
    def validate_contexts(cls, v):
        if not v:
            raise ValueError("Contexts cannot be empty")
        return v

    @validator("meta")
    def validate_meta(cls, v):
        required_keys = [
            "topic",
            "difficulty",
            "retrieval_k",
            "answer_model",
            "timestamp",
        ]
        missing_keys = [key for key in required_keys if key not in v]
        if missing_keys:
            raise ValueError(f"Missing required meta keys: {missing_keys}")
        return v


class TestsetDataFrame(BaseModel):
    """Schema for the complete testset dataframe."""

    data: List[TestsetRow]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([row.dict() for row in self.data])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TestsetDataFrame":
        """Create from pandas DataFrame."""
        rows = []
        for _, row in df.iterrows():
            rows.append(TestsetRow(**row.to_dict()))
        return cls(data=rows)


class PredictionDataFrame(BaseModel):
    """Schema for the complete predictions dataframe."""

    data: List[PredictionRow]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([row.dict() for row in self.data])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "PredictionDataFrame":
        """Create from pandas DataFrame."""
        rows = []
        for _, row in df.iterrows():
            rows.append(PredictionRow(**row.to_dict()))
        return cls(data=rows)


def validate_testset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate testset dataframe against schema.

    Args:
        df: Input dataframe

    Returns:
        Validated dataframe

    Raises:
        ValueError: If validation fails
    """
    try:
        # Validate the dataframe
        testset = TestsetDataFrame.from_dataframe(df)
        print(f"Testset validation passed: {len(testset.data)} rows")
        return testset.to_dataframe()
    except Exception as e:
        raise ValueError(f"Testset validation failed: {str(e)}")


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate predictions dataframe against schema.

    Args:
        df: Input dataframe

    Returns:
        Validated dataframe

    Raises:
        ValueError: If validation fails
    """
    try:
        # Validate the dataframe
        predictions = PredictionDataFrame.from_dataframe(df)
        print(f"Predictions validation passed: {len(predictions.data)} rows")
        return predictions.to_dataframe()
    except Exception as e:
        raise ValueError(f"Predictions validation failed: {str(e)}")


def create_empty_testset() -> pd.DataFrame:
    """Create an empty testset dataframe with correct schema."""
    return pd.DataFrame(columns=["question", "ground_truth", "topic", "difficulty"])


def create_empty_predictions() -> pd.DataFrame:
    """Create an empty predictions dataframe with correct schema."""
    return pd.DataFrame(
        columns=["question", "ground_truth", "contexts", "answer", "meta"]
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the schemas
    print("🧪 Testing dataset schemas...")

    # Test testset validation
    test_data = [
        {
            "question": "What properties are available in Manchester?",
            "ground_truth": "Properties in Manchester include...",
            "topic": "search",
            "difficulty": "easy",
        }
    ]

    test_df = pd.DataFrame(test_data)
    try:
        validated_df = validate_testset(test_df)
        print("Testset validation test passed")
    except Exception as e:
        print(f"Testset validation test failed: {e}")

    print("Schema validation tests completed")
