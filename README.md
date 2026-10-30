# NoteBot AI – AI-Powered PDF Study Assistant

An intelligent PDF study assistant built using Python, Streamlit, Google Gemini AI, LangChain, and FAISS.

NoteBot AI allows users to upload PDF documents, ask questions, generate summaries, take AI-generated quizzes, and explore document analytics through an interactive web application.

## Features

- **PDF Upload:** Upload PDF documents for analysis.
- **AI-Powered Chat:** Ask questions about uploaded PDF documents and receive contextual answers.
- **Source Citations:** View relevant PDF page numbers and source passages.
- **Conversation Memory:** Understand follow-up questions using previous conversation context.
- **PDF Summarization:** Generate structured summaries with important concepts and takeaways.
- **AI Quiz Generator:** Generate multiple-choice questions based on PDF content.
- **OCR Support:** Extract text from scanned PDF pages using Tesseract OCR.
- **Document Analytics:** View page counts, extracted characters, indexed chunks, OCR information, and processing time.
- **Vector Search:** Retrieve relevant document passages using FAISS.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Google Gemini AI | AI-powered responses and summaries |
| LangChain | LLM integration and text processing |
| FAISS | Vector database and similarity search |
| PyPDF2 | PDF text extraction |
| Tesseract OCR | Text recognition from scanned pages |
| Poppler | PDF page conversion |
| Google Generative AI Embeddings | Document embeddings |

## Project Structure

```text
GenAIChatBot/
│
├── .streamlit/
│   └── secrets.toml
│
├── saved_indexes/
│
├── MyChatBot.py
├── check_models.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd GenAIChatBot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Create a file named `secrets.toml` inside the `.streamlit` folder.

Add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Never share or upload your actual API key.

### 5. Configure Tesseract OCR and Poppler

Install Tesseract OCR and Poppler on your computer.

Update their installation paths in `MyChatBot.py` according to your system.

### 6. Run the application

```bash
streamlit run MyChatBot.py
```

The application will open in your web browser.

## How It Works

1. The user uploads a PDF document.
2. Text is extracted from the PDF using PyPDF2.
3. Scanned pages are processed using OCR when necessary.
4. Extracted text is divided into smaller chunks.
5. Google Generative AI Embeddings convert the chunks into vector representations.
6. FAISS stores the vectors for similarity-based retrieval.
7. LangChain connects the retrieved information with Google Gemini AI.
8. The application generates answers, summaries, and quizzes based on the document content.

## Project Objective

The main objective of NoteBot AI is to make studying and understanding PDF documents easier through artificial intelligence.

It combines document processing, natural language understanding, semantic search, and interactive learning features in one application.

## Developed Using

Python | Streamlit | Google Gemini AI | LangChain | FAISS

---

**Note:** This project requires a Gemini API key and locally configured OCR dependencies to run all features.