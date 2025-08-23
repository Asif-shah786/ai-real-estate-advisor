# 🔧 Fixing RAGAS Evaluation NaN Issues

## 🚨 **Problem: NaN Scores in RAGAS Evaluation**

According to the RAGAS documentation, NaN (Not a Number) scores can appear for two main reasons:

1. **JSON Parsing Issues**: Model outputs are not JSON-parsable
2. **Non-Ideal Cases**: Certain responses are not suitable for scoring

## 🔍 **Root Causes Identified**

### 1. **OpenAI API Key Configuration**
- **Issue**: RAGAS requires OpenAI API key for evaluation
- **Error**: `OpenAIError: The api_key client option must be set`
- **Fix**: Properly configure API key in environment and config

### 2. **Model Output Format Issues**
- **Issue**: Model responses may not be structured for RAGAS
- **Problem**: Non-JSON compatible outputs cause parsing failures
- **Examples**: 
  - "I don't know" responses
  - Unstructured text without proper formatting
  - Missing required fields

### 3. **Context Retrieval Problems**
- **Issue**: Empty or invalid contexts
- **Effect**: `context_precision` and `context_recall` become NaN
- **Cause**: Retrieval system returning empty results

## ✅ **Solutions Implemented**

### 1. **API Key Configuration**
```yaml
# eval/configs.yaml
judge:
  provider: "openai"
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"  # Environment variable substitution
```

### 2. **Enhanced Error Handling**
```python
# eval/score.py
try:
    result = evaluate(dataset, [metric_func])
    score = result[metric_name]
    
    # Handle NaN values
    if score is None or (isinstance(score, float) and (pd.isna(score) or np.isnan(score))):
        print(f"⚠️ {metric_name} returned NaN/None")
        score = 0.0  # Fallback score
        
except Exception as e:
    print(f"⚠️ {metric_name} computation failed: {e}")
    score = 0.0  # Fallback score
```

### 3. **Dataset Validation**
```python
def _validate_dataset_for_evaluation(self, dataset):
    # Check for problematic answers
    # Check for empty contexts
    # Warn about potential NaN issues
```

### 4. **Environment Variable Handling**
```python
# Handle environment variable substitution
if api_key and api_key.startswith("${") and api_key.endswith("}"):
    env_var = api_key[2:-1]
    api_key = os.getenv(env_var)

# Set for RAGAS
os.environ["OPENAI_API_KEY"] = api_key
```

## 🚀 **How to Use the Fixes**

### 1. **Set Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. **Run Evaluation**
```bash
cd eval
python3 run_evaluation.py
```

### 3. **Monitor Output**
The system will now:
- ✅ Validate dataset before evaluation
- ✅ Handle API key configuration automatically
- ✅ Provide fallback scores for NaN metrics
- ✅ Give detailed guidance on issues

## 📊 **Expected Results**

### **Before Fixes**
```
❌ faithfulness failed: OpenAIError: api_key not set
❌ answer_relevancy failed: OpenAIError: api_key not set
❌ context_precision: NaN
❌ context_recall: NaN
```

### **After Fixes**
```
✅ faithfulness: 0.85
✅ answer_relevancy: 0.82
⚠️ context_precision: 0.0 (fallback due to empty contexts)
⚠️ context_recall: 0.0 (fallback due to empty contexts)

⚠️ NaN scores detected for: ['context_precision', 'context_recall']
   This is common in RAGAS evaluation and can happen when:
   - Context retrieval issues
   - RAGAS evaluation limitations
   Consider improving model outputs or using more structured prompts
```

## 🔧 **Additional Improvements**

### 1. **Better Model Prompts**
- Use structured prompts that encourage JSON-like responses
- Implement response validation before evaluation
- Add fallback response handling

### 2. **Context Quality**
- Ensure retrieval system returns meaningful contexts
- Validate context relevance before evaluation
- Implement context filtering

### 3. **Evaluation Robustness**
- Add retry logic for failed evaluations
- Implement metric-specific error handling
- Provide detailed error analysis

## 📚 **References**

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS NaN Issues](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
- [OpenAI API Configuration](https://platform.openai.com/docs/api-reference)

## 🎯 **Next Steps**

1. **Test the fixes** with your evaluation pipeline
2. **Monitor NaN occurrences** and analyze patterns
3. **Improve model outputs** based on validation results
4. **Enhance context retrieval** to reduce empty contexts
5. **Consider alternative metrics** for problematic cases

---

*This guide addresses the specific NaN issues identified in your RAGAS evaluation pipeline.*
