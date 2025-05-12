
# 🔍 RAG-Based Document QA System

This project is a Retrieval-Augmented Generation (RAG) system built with FAISS and HuggingFace Transformers. It enables document ingestion, semantic indexing, search (standard and diverse), and free-text answer generation using local LLMs.

---

## 🛠️ Environment Setup

### 1. Create and activate a virtual environment
```bash
python -m venv rag_env
.
ag_env\Scriptsctivate  # On Windows
# or
source rag_env/bin/activate  # On macOS/Linux
```

### 2. Install all dependencies
```bash
pip install -r requirements.txt

# Or install individually:
pip install langchain sentence-transformers faiss-cpu PyMuPDF python-docx scikit-learn transformers accelerate
```

---

## 📁 Project Structure

```
rag-assignment/
├── documents/              # Source files (.pdf, .docx, .txt)
├── faiss_store/            # Vector index & metadata
├── main.py                 # Main program logic
└── README.md               # Project documentation
```

---

## 🧠 System Workflow Overview

1. **Document Ingestion**: PDFs, DOCX, and TXT files are parsed and loaded.
2. **Text Chunking**: Each file is split into manageable text segments.
3. **Embedding Generation**: Sentence embeddings created using `all-MiniLM-L6-v2`.
4. **FAISS Indexing**: Vectors are stored in a fast FAISS index.
5. **Search Options**:
   - Mode 2: Simple semantic top-K
   - Mode 3: MMR search for diversity
6. **Answer Generation**:
   - Mode 4 uses HuggingFace’s `google/flan-t5-base` for text generation

---

## 🧪 Retrieval Strategy Comparison

| Mode | Name           | Description                             |
|------|----------------|-----------------------------------------|
| 2    | Semantic Search| Standard top-K retrieval                |
| 3    | MMR Search     | Improves diversity of top-K documents   |
| 4    | RAG Answer     | Synthesizes answer using retrieved docs |

---

## 📊 Evaluation (Qualitative)

- **Precision**: High for semantic retrieval
- **Diversity**: MMR helps reduce redundancy
- **Answer Accuracy**: Reasonable for flan-t5-base with good context
- **Speed**: Near-instant response with FAISS indexing

---

## ✅ Pros

- No need for OpenAI API or paid keys
- Lightweight LLM support using HuggingFace
- Clean modular implementation
- Works with local documents

---

## ⚠️ Limitations

- Limited generation detail with smaller models
- No automated scoring (F1/ROUGE/etc.)
- CLI only (no web UI)

---

## 🚧 Challenges Addressed

| Issue                                   | Fix Applied                             |
|----------------------------------------|------------------------------------------|
| API quota limits from OpenAI           | Switched to HuggingFace transformers     |
| LangChain embedding compatibility      | Used correct HuggingFace wrapper         |
| No chunk preview in results            | Stored chunk text in metadata            |
| Retrieval duplicates                   | MMR added to enhance result diversity    |

---

## 📥 Document Corpus Instructions

Simply place your `.pdf`, `.docx`, or `.txt` files into the `documents/` directory. The system auto-processes everything in that folder.

---

## 💽 Model + Index Files

| File                                      | Purpose                      |
|-------------------------------------------|-------------------------------|
| `index_miniLM.index`                      | FAISS vector index            |
| `index_miniLM_metadata.pkl`               | Metadata for each chunk       |

---

## ▶️ Running the System

```bash
python main.py
```

Choose a mode:
- `1`: Build FAISS index from documents
- `2`: Semantic search
- `3`: MMR search
- `4`: RAG-style answer generation

---

## 📌 Sample Query

```text
what is the difference between RAG-sequence and RAG-token models
```

---

## 🤝 Contributors

This version was prepared as a student submission using open-source tools. No external APIs or keys are required.

