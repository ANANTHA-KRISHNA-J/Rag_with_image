# Rag_with_image

A RAG (Retrieval-Augmented Generation) chatbot that lets you chat with **multiple PDFs, images containing text, and web links**. Built with LangChain, FAISS, Google Gemini, and a Streamlit front end.

## Features

- **Multiple input types**: PDFs, images with text (OCR), and web page links
- **Semantic chunking**: splits text by meaning instead of fixed size
- **Hybrid retrieval**: combines FAISS vector search (70%) with BM25 keyword search (30%)
- **Follow-up friendly**: rewrites follow-up questions into standalone queries using chat history
- **Short answers**: Gemini replies in 3 sentences or fewer
- **Model fallback**: switches to a backup Gemini model automatically on a 503 error
- **Error explainer**: unexpected Python errors are explained in plain language

## How it works

1. **Load**: text is extracted from PDFs (PyMuPDF), images (Tesseract OCR), or web pages (`WebBaseLoader`)
2. **Chunk**: `SemanticChunker` splits the text using `all-MiniLM-L6-v2` embeddings
3. **Index**: chunks are stored in a FAISS vector database
4. **Retrieve**: an ensemble of FAISS + BM25 finds the most relevant chunks
5. **Answer**: the question and retrieved chunks go to Gemini, which writes the final answer

## Project structure

| File | Purpose |
|------|---------|
| `backbone_funcv.py` | Core functions: loading, chunking, vector DB, retrieval, LLM calls |
| `fustv.py` | Streamlit app |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages (e.g. Tesseract for OCR) |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/ANANTHA-KRISHNA-J/Rag_with_image.git
cd Rag_with_image
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Install Tesseract OCR** (needed for reading text from images)
- Windows: download the installer from the [Tesseract releases page](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

**4. Add your Gemini API key**

Create a `.env` file in the project folder:
```
GEMINI_API_KEY=your_api_key_here
```

**5. Run the app**
```bash
streamlit run fustv.py
```

## Tech stack

- Python, Streamlit
- LangChain (community, experimental, HuggingFace)
- FAISS, BM25
- Sentence-Transformers (`all-MiniLM-L6-v2`)
- Google Gemini (`google-genai`)
- PyMuPDF, Pytesseract, Pillow

## Author

Made by [Ananthakrishna](https://github.com/ANANTHA-KRISHNA-J)
