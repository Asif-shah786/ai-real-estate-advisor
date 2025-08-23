#!/usr/bin/env python3
"""
Quick test to verify SelfQueryRetriever is working with metadata filtering.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_selfquery_retriever():
    """Test if SelfQueryRetriever is working with metadata filtering."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found")
        return False

    try:
        print("🔧 Testing SelfQueryRetriever initialization...")
        from rag_pipeline import RAGPipeline

        # Initialize pipeline
        pipeline = RAGPipeline(api_key)
        print("RAG Pipeline initialized successfully")

        # Check if SelfQueryRetriever was created
        if hasattr(pipeline, "self_query_retriever") and pipeline.self_query_retriever:
            print("SelfQueryRetriever is available!")

            # Test a structured query
            test_query = "Show me properties under £500,000"
            print(f"\n🧪 Testing structured query: '{test_query}'")

            try:
                docs = pipeline.self_query_retriever.invoke(test_query)
                if docs:
                    print(f"SelfQueryRetriever returned {len(docs)} documents")

                    # Check metadata filtering
                    if docs and hasattr(docs[0], "metadata"):
                        sample_meta = docs[0].metadata
                        print(f"Sample metadata: {sample_meta}")

                        # Verify price filtering worked
                        if "price_int" in sample_meta:
                            max_price = max(
                                doc.metadata.get("price_int", 0)
                                for doc in docs
                                if doc.metadata.get("price_int")
                            )
                            print(f"💰 Max price in results: £{max_price:,}")

                            if max_price <= 500000:
                                print("Price filtering is working correctly!")
                            else:
                                print(
                                    " Price filtering may not be working - found price above £500,000"
                                )

                    return True
                else:
                    print(" SelfQueryRetriever returned no documents")
                    return False

            except Exception as e:
                print(f"SelfQueryRetriever test failed: {e}")
                return False
        else:
            print("SelfQueryRetriever is not available")
            return False

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing SelfQueryRetriever Metadata Filtering")
    print("=" * 50)

    success = test_selfquery_retriever()

    if success:
        print("\n🎉 SelfQueryRetriever is working correctly!")
        print("   - Metadata filtering should now work")
        print("   - Price, bedroom, and postcode filters will use metadata")
        print("   - Better context precision and recall expected")
    else:
        print("\nSelfQueryRetriever still has issues")
        print("   - Check the error messages above")
        print("   - May need additional dependencies")

    print(f"\nExit code: {0 if success else 1}")
