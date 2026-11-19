# Personal AI Knowledge Assistant using Retrieval-Augmented Generation (RAG)

An AI-powered knowledge assistant that allows users to upload documents and ask questions using natural language. The application retrieves relevant information from a knowledge base and uses Google's Gemini API to generate context-aware responses.

## Project Overview

The Personal AI Knowledge Assistant uses Retrieval-Augmented Generation (RAG) to provide answers grounded in the content of uploaded documents and an existing machine learning knowledge base.

Instead of relying only on an LLM's pre-trained knowledge, the system retrieves relevant document chunks and supplies them as context when generating responses.

## Key Features

- Upload and process PDF, DOCX, TXT, CSV, and XLSX documents.
- Extract text and split documents into manageable chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store and search document embeddings using FAISS.
- Retrieve relevant information from the original and uploaded knowledge bases.
- Generate context-aware answers using the Google Gemini API.
- Display source information for retrieved content.
- Maintain conversational history and support follow-up questions.
- Reject questions when relevant information cannot be retrieved with sufficient similarity.
- Measure retrieval, response-generation, and total processing time.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Large Language Model | Google Gemini API |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Data Processing | Pandas, NumPy |
| Document Processing | PyPDF, python-docx, OpenPyXL |
| Text Chunking | LangChain Text Splitters |
| Environment Management | python-dotenv |

## System Architecture

The application follows this workflow:

1. **Document Ingestion:** Load documents from supported file formats.
2. **Text Extraction:** Extract text and relevant document metadata.
3. **Text Chunking:** Divide documents into smaller, meaningful text chunks.
4. **Embedding Generation:** Convert text chunks into numerical vector representations.
5. **Vector Storage:** Store embeddings in a FAISS index.
6. **Query Processing:** Process user questions and contextual follow-up questions.
7. **Semantic Retrieval:** Retrieve relevant document chunks using similarity search.
8. **Prompt Construction:** Combine the user question, conversation history, and retrieved context.
9. **Response Generation:** Use the Gemini API to generate a context-aware answer.
10. **Response Display:** Present the answer, source information, and processing time through Streamlit.

## Project Structure

```text
GUVI_my_sixth_project/
│
├── dashboard/
│   └── app.py
│
├── src/
│   ├── chunking.py
│   ├── conversation.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── rag_pipeline.py
│   └── retrieval.py
│
├── data/
│   └── processed/
│
├── vectorstore/
│
├── backups/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone [https://github.com/Jeshmitha03/AIML_Personal-AI-Knowledge-Assistant-using-RAG](https://github.com/Jeshmitha03/AIML_Personal-AI-Knowledge-Assistant-using-RAG)
cd GUVI_my_sixth_project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root using `.env.example` as a template.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Never upload your actual `.env` file or API key to GitHub.

### 5. Run the application

```bash
python -m streamlit run dashboard/app.py
```

The application will open in your browser.

## Supported Document Formats

- PDF
- DOCX
- TXT
- CSV
- XLSX

## Knowledge Base

The project includes a machine learning and deep learning question-answer knowledge base sourced from the Hugging Face dataset:

[Koushim/qa-ml-dl-jsonl](https://huggingface.co/datasets/Koushim/qa-ml-dl-jsonl)

The application uses Sentence Transformers to generate embeddings and FAISS for semantic similarity search.

## Potential Applications

- Personal document assistant
- Educational knowledge assistant
- Enterprise document search
- Technical documentation assistant
- Internal knowledge management

## Future Enhancements

- Support additional document formats.
- Improve retrieval using hybrid search and reranking.
- Add document-level access control.
- Explore evaluation techniques for retrieval quality and answer faithfulness.
- Deploy the application to a cloud platform.

## Author

**Jes​​hmitha J.**

GitHub: [Jeshmitha03](https://github.com/Jeshmitha03)

LinkedIn: [Jes​​hmitha J.](https://www.linkedin.com/in/jeshmitha03)

---

*This project was developed as part of an AI/ML learning and portfolio development journey.*