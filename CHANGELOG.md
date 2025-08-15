# Project Changelog

## 2025-08-15: Preprocessing Pipeline & JSONL Embedding Implementation

### 🎯 **What We Did**

1. **Removed V1 Code** - Eliminated old pandas dataframe agent implementation
2. **Implemented Preprocessing Pipeline** - Created structured dataset from raw CSV
3. **Added JSONL-Based Embedding** - Replaced CSV loading with pre-processed JSONL

### 📊 **Preprocessing Pipeline**

**File**: `preprocessing.py`
- **Input**: `manchester_properties_for_sale_mini.csv` (raw data)
- **Output**: 
  - `structured_properties.parquet` (49KB, 151 rows)
  - `embedding_docs.jsonl` (513KB, 487 lines)
  - `structured_properties.csv` (53KB, 151 rows)

**Improvements**:
- Added 40+ structured fields (amenities, price bands, location data)
- Implemented deduplication using `canonical_id`
- Created sentence-window chunks for better embeddings
- Extracted city, postcode, geohash from addresses

### 🔄 **App Changes**

**File**: `app.py`
- **Added**: `load_docs_from_jsonl()` method
- **Modified**: `setup_vectordb()` to prioritize JSONL over CSV
- **Updated**: Data source display in sidebar
- **Fixed**: Source reference display for JSONL metadata structure
- **Preserved**: All existing CSV and URL functionality

### 📁 **File Usage**

| File                                      | Purpose                                   | Why We Use It                             |
| ----------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| `embedding_docs.jsonl`                    | **Primary** - Main dataset for embeddings | Pre-processed, clean text, rich metadata  |
| `structured_properties.csv`               | **Fallback** - If JSONL fails             | Structured data with all extracted fields |
| `manchester_properties_for_sale_mini.csv` | **Raw** - Original data                   | Source for preprocessing pipeline         |

### ✅ **Benefits**

- **Better embeddings** with clean, structured text
- **Rich metadata** for filtering and search
- **Faster startup** - no CSV processing during app init
- **Improved RAG performance** with pre-chunked data
- **Enhanced user experience** with detailed property insights

### 🔧 **Technical Details**

- **Embedding strategy**: Sentence-window chunks + whole document
- **Metadata preservation**: All 40+ fields attached to vectors
- **Fallback system**: JSONL → CSV → External URLs
- **No breaking changes**: All existing functionality preserved

---
*Last updated: 2025-08-15*
