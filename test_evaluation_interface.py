#!/usr/bin/env python3
"""
Test script to verify the evaluation interface works correctly.

This script tests the clean RAG pipeline interface that evaluation code expects:
pipeline.run_query(query) => {answer: str, contexts: List[str], meta: dict}
"""

import os
from rag_pipeline import create_rag_pipeline


def test_evaluation_interface():
    """Test the evaluation interface."""
    print("🧪 Testing Evaluation Interface...\n")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set OPENAI_API_KEY before running this test")
        return False
    
    try:
        # Create the RAG pipeline
        print("1. Creating RAG Pipeline...")
        pipeline = create_rag_pipeline(api_key)
        
        if pipeline is None:
            print("❌ Failed to create RAG pipeline")
            return False
        
        print("✅ RAG Pipeline created successfully")
        
        # Test the evaluation interface
        print("\n2. Testing run_query method...")
        
        # Test query 1: Simple property search
        print("\n   Testing: 'Show me properties in Manchester'")
        result1 = pipeline.run_query("Show me properties in Manchester")
        
        print(f"   ✅ Answer: {result1['answer'][:100]}...")
        print(f"   ✅ Contexts: {len(result1['contexts'])} retrieved")
        print(f"   ✅ Meta: {result1['meta']['source_count']} sources")
        
        # Test query 2: Contextual question
        print("\n   Testing: 'What is the crime rate in the second property?'")
        result2 = pipeline.run_query("What is the crime rate in the second property?")
        
        print(f"   ✅ Answer: {result2['answer'][:100]}...")
        print(f"   ✅ Contexts: {len(result2['contexts'])} retrieved")
        print(f"   ✅ Meta: {result2['meta']['source_count']} sources")
        
        # Test query 3: Legal question
        print("\n   Testing: 'What legal documents do I need for property purchase?'")
        result3 = pipeline.run_query("What legal documents do I need for property purchase?")
        
        print(f"   ✅ Answer: {result3['answer'][:100]}...")
        print(f"   ✅ Contexts: {len(result3['contexts'])} retrieved")
        print(f"   ✅ Meta: {result3['meta']['source_count']} sources")
        
        # Verify the interface structure
        print("\n3. Verifying Interface Structure...")
        
        required_keys = ["answer", "contexts", "meta"]
        meta_keys = ["source_count", "retriever_type", "timestamp", "query", "source_metadata"]
        
        for key in required_keys:
            if key not in result1:
                print(f"❌ Missing required key: {key}")
                return False
        
        for key in meta_keys:
            if key not in result1["meta"]:
                print(f"❌ Missing meta key: {key}")
                return False
        
        print("✅ Interface structure verified correctly")
        
        # Test pipeline info
        print("\n4. Testing Pipeline Info...")
        info = pipeline.get_pipeline_info()
        print(f"   ✅ Retriever: {info['retriever_type']}")
        print(f"   ✅ Memory: {info['memory_type']}")
        print(f"   ✅ QA Chain: {info['qa_chain_type']}")
        print(f"   ✅ Vector DB: {info['vectordb_type']}")
        
        # Test memory clearing
        print("\n5. Testing Memory Management...")
        pipeline.clear_memory()
        print("✅ Memory cleared successfully")
        
        # Test without memory
        print("\n6. Testing Query Without Memory...")
        result_no_memory = pipeline.run_query("Show me properties in Manchester", use_memory=False)
        print(f"   ✅ Query without memory: {result_no_memory['answer'][:100]}...")
        
        print("\n🎉 All Evaluation Interface Tests Passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_evaluation_interface()
    if success:
        print("\n✅ The RAG pipeline is ready for evaluation!")
        print("   Interface: pipeline.run_query(query) => {answer, contexts, meta}")
        print("   Streamlit UI: Cleanly separated from RAG logic")
    else:
        print("\n❌ Evaluation interface test failed")
        print("   Please check the errors above")
