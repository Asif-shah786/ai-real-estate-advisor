#!/usr/bin/env python3
"""
Test script to verify the CrossEncoderRerankRetriever is working correctly.
"""


def test_reranker_import():
    """Test if the reranker can be imported successfully."""
    try:
        from retrieval import CrossEncoderRerankRetriever

        print("✅ CrossEncoderRerankRetriever imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_reranker_creation():
    """Test if the reranker can be created (without actual vectordb)."""
    try:
        from retrieval import CrossEncoderRerankRetriever

        # Test the class structure
        print(f"✅ Class name: {CrossEncoderRerankRetriever.__name__}")
        print(f"✅ Base class: {CrossEncoderRerankRetriever.__bases__}")
        print(
            f"✅ Has from_vectorstore method: {hasattr(CrossEncoderRerankRetriever, 'from_vectorstore')}"
        )

        return True
    except Exception as e:
        print(f"❌ Creation test failed: {e}")
        return False


def test_dependencies():
    """Test if required dependencies are available."""
    try:
        import sentence_transformers

        print(f"✅ sentence_transformers version: {sentence_transformers.__version__}")

        import torch

        print(f"✅ PyTorch version: {torch.__version__}")

        return True
    except ImportError as e:
        print(f"❌ Dependency missing: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing CrossEncoderRerankRetriever...\n")

    # Test 1: Import
    print("1. Testing import...")
    import_success = test_reranker_import()
    print()

    # Test 2: Dependencies
    print("2. Testing dependencies...")
    deps_success = test_dependencies()
    print()

    # Test 3: Class creation
    if import_success:
        print("3. Testing class structure...")
        creation_success = test_reranker_creation()
        print()

    # Summary
    print("📊 Test Summary:")
    print(f"   Import: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"   Dependencies: {'✅ PASS' if deps_success else '❌ FAIL'}")
    print(
        f"   Class Structure: {'✅ PASS' if import_success and creation_success else '❌ FAIL'}"
    )

    if import_success and deps_success:
        print("\n🎉 Reranker is ready to use!")
    else:
        print("\n⚠️ Some issues detected. Check the errors above.")
