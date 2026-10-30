import streamlit as st
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from io import BytesIO

import pytesseract
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS
logger = logging.getLogger(__name__)

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="NoteBot AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 NoteBot AI")
st.caption(
    "Your PDF study assistant with citations, summaries, quizzes, "
    "conversation memory, and document analytics."
)

# ==========================================================
# 2. TESSERACT AND POPPLER CONFIGURATION
# ==========================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
POPPLER_PATH = (
    r"C:\Users\Divya p\Downloads\Release-26.09.0-0"
    r"\poppler-26.09.0\Library\bin"
)
# ==========================================================
# 3. INDEX STORAGE
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "saved_indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 4. API KEY
# ==========================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

except (KeyError, FileNotFoundError):
    st.error(
        "Gemini API key not found. Add GEMINI_API_KEY "
        "to .streamlit/secrets.toml."
    )
    st.stop()

# ==========================================================
# 5. SESSION STATE
# ==========================================================

defaults = {
    "chat_history": [],
    "active_pdf_hash": None,
    "loaded_pdf_hash": None,
    "loaded_vector_store": None,
    "pdf_stats": {},
    "summary_cache": {},
    "quiz_questions": [],
    "quiz_answers": {},
    "quiz_checked": False,
}

for state_key, default_value in defaults.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# ==========================================================
# 6. GEMINI MODELS
# ==========================================================

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        max_output_tokens=1800,
    )


@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )


def response_to_text(response):
    """Convert a Gemini response into readable text."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []

        for content_item in content:
            if isinstance(content_item, str):
                parts.append(content_item)

            elif (
                isinstance(content_item, dict)
                and content_item.get("type") == "text"
            ):
                parts.append(content_item.get("text", ""))

        return "\n".join(parts)

    return str(content)

# ==========================================================
# 7. EXTRACT PDF TEXT, INCLUDING SCANNED PAGES
# ==========================================================

def extract_pdf_text(file_bytes):
    """Return one extracted text string per PDF page."""
    reader = PdfReader(BytesIO(file_bytes))
    extracted_page_texts = []
    detected_ocr_pages = []

    for page_index, pdf_page in enumerate(reader.pages):
        page_number = page_index + 1
        try:
            extracted_text = pdf_page.extract_text() or ""

        except (ValueError, TypeError):
            extracted_text = ""

        # Use OCR when the page has little extractable text.
        if len(extracted_text.strip()) < 40:

            try:
                images = convert_from_bytes(
                    file_bytes,
                    first_page=page_number,
                    last_page=page_number,
                    dpi=200,
                    poppler_path=POPPLER_PATH,
                )

                if images:
                    extracted_text = pytesseract.image_to_string(
                        images[0]
                    )

                    detected_ocr_pages.append(page_number)

            except Exception as ocr_error:
                st.warning(
                    f"OCR failed on page {page_number}: {ocr_error}"
                )

        extracted_page_texts.append(extracted_text.strip())

    return extracted_page_texts, detected_ocr_pages

# ==========================================================
# 8. CHUNK TEXT WHILE PRESERVING PAGE NUMBERS
# ==========================================================

def create_page_documents(page_texts):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    documents = []

    for page_number, page_text in enumerate(
        page_texts,
        start=1,
    ):

        if not page_text.strip():
            continue

        page_documents = splitter.create_documents(
            [page_text],
            metadatas=[{"page": page_number}],
        )

        documents.extend(page_documents)

    return documents

# ==========================================================
# 9. CREATE OR LOAD FAISS INDEX
# ==========================================================

def process_pdf(uploaded_pdf_bytes):

    document_hash = hashlib.sha256(
        uploaded_pdf_bytes
    ).hexdigest()

    index_path = INDEX_DIR / (
        f"{document_hash}_page_citations_v1"
    )

    stats_path = INDEX_DIR / f"{document_hash}_stats.json"

    embeddings = get_embeddings()

    # Load existing FAISS index
    if index_path.exists():

        try:
            loaded_vector_store = FAISS.load_local(
                str(index_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

            loaded_document_stats = {}

            if stats_path.exists():

                with open(
                    stats_path,
                    "r",
                    encoding="utf-8",
                ) as stats_file:

                    loaded_document_stats = json.load(stats_file)

            return (
                loaded_vector_store,
                document_hash,
                loaded_document_stats,
            )

        except Exception as index_error:

            st.warning(
                "Saved index could not be loaded. "
                f"Rebuilding it: {index_error}"
            )

    # Extract PDF text
    extracted_page_texts, detected_ocr_pages = extract_pdf_text(
        uploaded_pdf_bytes
    )

    documents = create_page_documents(
        extracted_page_texts
    )

    if not documents:
        raise ValueError(
            "No readable text was found. "
            "Check the PDF and OCR setup."
        )

    # Create FAISS vector database
    created_vector_store = FAISS.from_documents(
        documents,
        embeddings,
    )

    created_vector_store.save_local(
        str(index_path)
    )

    generated_document_stats = {
        "page_count": len(extracted_page_texts),

        "pages_with_text": sum(
            bool(page_text.strip())
            for page_text in extracted_page_texts
        ),

        "total_characters": sum(
            len(page_text)
            for page_text in extracted_page_texts
        ),

        "ocr_pages": detected_ocr_pages,

        "chunk_count": len(documents),

        "processing_time": None,
    }

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as stats_file:

        json.dump(
            generated_document_stats,
            stats_file,
            indent=2,
        )

    return (
        created_vector_store,
        document_hash,
        generated_document_stats,
    )

# ==========================================================
# 10. FOLLOW-UP QUESTION REWRITER
# FEATURE 3
# ==========================================================

def rewrite_followup_question(question, chat_history):

    if not chat_history:
        return question

    recent_messages = chat_history[-6:]

    history_text = "\n".join(
        f"{chat_message['role']}: {chat_message['content']}"
        for chat_message in recent_messages
    )

    prompt = f"""
Rewrite the current user question as a standalone search query
for a PDF retrieval system.

Use the conversation to resolve references such as:
it, they, this, that, the second point, and explain more.

Do not answer the question.
Do not add facts that are not in the conversation.
Return only the rewritten search query.

CONVERSATION:
{history_text}

CURRENT QUESTION:
{question}
"""

    try:
        response = get_llm().invoke(prompt)

        rewritten = response_to_text(response).strip()

        return rewritten or question

    except Exception as rewrite_error:
        # Fall back to the original question, but keep a record of why.
        logger.warning(
            "Follow-up rewrite failed, using original question: %s",
            rewrite_error,
        )
        return question


# ==========================================================
# 11. CHECK WHETHER PDF CONTEXT SUPPORTS THE QUESTION
# FEATURE 6
# ==========================================================

def check_context_support(question, context):

    prompt = f"""
Determine whether the PDF passages below contain enough relevant
information to answer the question.

Reply with exactly YES or NO.
Reply NO if the passages are unrelated or insufficient.

QUESTION:
{question}

PDF PASSAGES:
{context}
"""

    response = get_llm().invoke(prompt)

    verdict = response_to_text(response).strip().upper()

    return verdict.startswith("YES")


# ==========================================================
# 12. ANSWER A QUESTION USING PDF PASSAGES
# ==========================================================

def answer_question(question, vector_store, chat_history):

    standalone_query = rewrite_followup_question(
        question,
        chat_history,
    )

    source_documents = vector_store.similarity_search(
        standalone_query,
        k=4,
    )

    context = "\n\n".join(
        document.page_content
        for document in source_documents
    )

    if not context.strip():

        return (
            "I couldn't find relevant text in this PDF.",
            [],
            standalone_query,
        )

    # Check whether the retrieved passages support the question.
    if not check_context_support(question, context):

        return (
            "I couldn't find enough relevant information in the "
            "retrieved PDF passages to answer that confidently. "
            "Try asking the question differently or using more "
            "specific keywords.",

            [
                {
                    "page": document.metadata.get(
                        "page",
                        "Unknown",
                    ),
                    "text": document.page_content,
                }
                for document in source_documents
            ],

            standalone_query,
        )

    recent_history = chat_history[-6:]

    history_text = "\n".join(
        f"{chat_message['role'].upper()}: "
        f"{chat_message['content']}"
        for chat_message in recent_history
    )

    prompt = f"""
You are NoteBot, a helpful study tutor.

Answer using only the PDF passages supplied below.
Use the conversation to understand follow-up questions.
If the passages do not contain the answer, say so.
Do not invent facts, page numbers, quotations, or details.
Explain in simple language.

PREVIOUS CONVERSATION:
{history_text}

CURRENT QUESTION:
{question}

PDF PASSAGES:
{context}

Give a clear and useful answer.
"""

    response = get_llm().invoke(prompt)

    answer = response_to_text(response).strip()

    sources = [
        {
            "page": document.metadata.get(
                "page",
                "Unknown",
            ),
            "text": document.page_content,
        }
        for document in source_documents
    ]

    return answer, sources, standalone_query


# ==========================================================
# 13. SUMMARIZE PDF IN SECTIONS
# FEATURE 4
# ==========================================================

def summarize_pdf(page_texts):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=300,
    )

    full_text = "\n\n".join(
        f"PDF Page {page_number}:\n{page_text}"
        for page_number, page_text in enumerate(
            page_texts,
            start=1,
        )
        if page_text.strip()
    )

    if not full_text.strip():
        return "No readable text was found to summarize."

    sections = splitter.split_text(full_text)

    section_summaries = []

    for section_index, section in enumerate(
        sections,
        start=1,
    ):

        prompt = f"""
Summarize this section of a study PDF.

Keep the important concepts, definitions, processes, and facts.
Do not add information that is not in the section.
Use simple language and clear bullet points.

SECTION {section_index} OF {len(sections)}:
{section}
"""

        response = get_llm().invoke(prompt)

        section_summaries.append(
            response_to_text(response)
        )

    if len(section_summaries) == 1:
        return section_summaries[0]

    combined = "\n\n".join(section_summaries)

    final_prompt = f"""
Create one well-organized overall summary from these section summaries.

Include:
1. Main topic
2. Key concepts
3. Important definitions or processes
4. Final takeaways

Remove repeated points. Use only the information in these summaries.
Do not invent facts.

SECTION SUMMARIES:
{combined}
"""

    response = get_llm().invoke(final_prompt)

    return response_to_text(response)


# ==========================================================
# 14. GENERATE MULTIPLE-CHOICE QUIZ QUESTIONS
# FEATURE 5
# ==========================================================

def generate_quiz(vector_store, number_of_questions):

    documents = vector_store.similarity_search(
        "important concepts definitions key facts and main ideas",
        k=12,
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
Create exactly {number_of_questions} multiple-choice study questions
based only on the PDF passages below.

Return valid JSON only, with no Markdown fences, using this structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A",
      "explanation": "Short explanation supported by the PDF"
    }}
  ]
}}

Rules:
- Exactly four options per question.
- There must be one correct option.
- correct_answer must be A, B, C, or D.
- Do not create questions whose answers are absent from the passages.
- Do not repeat questions.
- Do not include answers outside the JSON.

PDF PASSAGES:
{context}
"""

    response = get_llm().invoke(prompt)

    raw_text = response_to_text(response).strip()

    # Remove Markdown code fences if Gemini includes them.
    raw_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw_text,
    )

    raw_text = re.sub(
        r"\s*```$",
        "",
        raw_text,
    )

    data = json.loads(raw_text)

    questions = data.get("questions", [])

    if not questions:
        raise ValueError(
            "Gemini did not return any quiz questions."
        )

    valid_questions = []

    for quiz_item in questions:

        options = quiz_item.get("options", {})

        correct = quiz_item.get(
            "correct_answer",
            "",
        ).upper()

        if (
            quiz_item.get("question")
            and all(
                option_key in options
                for option_key in ["A", "B", "C", "D"]
            )
            and correct in ["A", "B", "C", "D"]
        ):

            valid_questions.append(
                {
                    "question": quiz_item["question"],
                    "options": options,
                    "correct_answer": correct,
                    "explanation": quiz_item.get(
                        "explanation",
                        "",
                    ),
                }
            )

    if not valid_questions:
        raise ValueError(
            "The generated quiz format was invalid."
        )

    return valid_questions


# ==========================================================
# 15. SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📂 My Notes")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if st.button(
        "Clear chat history",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


# With accept_multiple_files=False Streamlit returns a single file
# (or None), but its type hints also allow a list. Normalise it so
# the type checker (and any future change) is handled safely.
if isinstance(uploaded_file, list):
    uploaded_file = uploaded_file[0] if uploaded_file else None


# ==========================================================
# 16. LOAD AND PROCESS PDF
# ==========================================================

if uploaded_file is None:

    st.info("Upload a PDF to begin.")

    st.stop()


pdf_bytes = uploaded_file.getvalue()

if not pdf_bytes:

    st.warning("The uploaded PDF is empty.")

    st.stop()


current_pdf_hash = hashlib.sha256(
    pdf_bytes
).hexdigest()


# Reset information when a different PDF is uploaded.
if st.session_state.active_pdf_hash != current_pdf_hash:

    st.session_state.active_pdf_hash = current_pdf_hash

    st.session_state.chat_history = []

    st.session_state.summary_cache = {}

    st.session_state.quiz_questions = []

    st.session_state.quiz_answers = {}

    st.session_state.quiz_checked = False


# Process the PDF only when necessary.
if st.session_state.loaded_pdf_hash != current_pdf_hash:

    try:

        with st.spinner(
            "Processing PDF and preparing search index..."
        ):

            processing_start_time = time.time()

            new_vector_store, new_pdf_hash, new_stats = process_pdf(
                pdf_bytes
            )

            processing_time = (
                time.time() - processing_start_time
            )

            new_stats["processing_time"] = round(
                processing_time,
                2,
            )

            st.session_state.loaded_vector_store = new_vector_store

            st.session_state.loaded_pdf_hash = new_pdf_hash

            st.session_state.pdf_stats = new_stats

    except Exception as pdf_process_error:

        st.error(
            f"Unable to process this PDF: {pdf_process_error}"
        )

        st.stop()


active_vector_store = st.session_state.loaded_vector_store

st.subheader(f"📄 {uploaded_file.name}")


# ==========================================================
# 17. NAVIGATION
# ==========================================================

tab_chat, tab_summary, tab_quiz, tab_analytics = st.tabs(
    [
        "💬 Chat",
        "📝 Summary",
        "🧠 Quiz",
        "📊 Analytics",
    ]
)


# ==========================================================
# 18. CHAT INTERFACE
# ==========================================================

with tab_chat:

    st.subheader("Ask questions about your PDF")

    # Display previous conversation.
    for history_item in st.session_state.chat_history:

        with st.chat_message(history_item["role"]):

            st.markdown(history_item["content"])

            if history_item["role"] == "assistant":

                saved_sources = history_item.get(
                    "sources",
                    [],
                )

                if saved_sources:

                    with st.expander("View PDF sources"):

                        for source_number, source_item in enumerate(
                            saved_sources,
                            start=1,
                        ):

                            st.markdown(
                                f"**Source {source_number} — "
                                f"PDF Page {source_item['page']}**"
                            )

                            st.write(source_item["text"])


    # Get new question.
    user_question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if user_question:

        with st.chat_message("user"):

            st.markdown(user_question)

        # Save the question.
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question,
            }
        )


        try:

            answer_start_time = time.time()

            with st.spinner(
                "Searching the PDF and preparing an answer..."
            ):

                # Exclude the current question from previous history.
                previous_history = (
                    st.session_state.chat_history[:-1]
                )

                ai_answer, ai_sources, _ = answer_question(
                    user_question,
                    active_vector_store,
                    previous_history,
                )


            answer_time = (
                time.time() - answer_start_time
            )

            with st.chat_message("assistant"):

                st.markdown(ai_answer)

                st.caption(
                    f"Response time: {answer_time:.2f} seconds"
                )


                if ai_sources:

                    with st.expander("View PDF sources"):

                        for source_number, source_item in enumerate(
                            ai_sources,
                            start=1,
                        ):

                            st.markdown(
                                f"**Source {source_number} — "
                                f"PDF Page {source_item['page']}**"
                            )

                            st.write(source_item["text"])


            # Save the answer.
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": ai_answer,
                    "sources": ai_sources,
                }
            )

        except Exception as chat_error:

            st.error(
                f"Unable to generate an answer: {chat_error}"
            )


# ==========================================================
# 19. AUTOMATIC SUMMARY
# FEATURE 4
# ==========================================================

with tab_summary:

    st.subheader("PDF Summary")

    st.write(
        "Generate a summary of the uploaded document, "
        "including important concepts and takeaways."
    )


    if st.button(
        "Generate PDF summary",
        key="generate_summary",
    ):

        cache_key = current_pdf_hash

        try:

            with st.spinner(
                "Reading the PDF and creating a summary..."
            ):

                # Extract PDF text.
                summary_page_texts, _ = extract_pdf_text(
                    pdf_bytes
                )

                summary = summarize_pdf(
                    summary_page_texts
                )

                st.session_state.summary_cache[
                    cache_key
                ] = summary

        except Exception as summary_error:

            st.error(
                f"Unable to generate the summary: {summary_error}"
            )


    # Display saved summary outside the button condition.
    cached_summary = st.session_state.summary_cache.get(
        current_pdf_hash
    )


    if (
        isinstance(cached_summary, str)
        and cached_summary.strip()
    ):

        st.markdown(cached_summary)

        st.download_button(
            "Download summary as TXT",
            data=cached_summary,
            file_name="notebot_summary.txt",
            mime="text/plain",
        )

    else:

        st.info(
            "Click 'Generate PDF summary' to create your summary."
        )

# ==========================================================
# 20. QUIZ GENERATOR
# FEATURE 5
# ==========================================================

with tab_quiz:
    st.subheader("Test your understanding")


    quiz_count = st.selectbox(
        "Number of questions",
        options=[3, 5, 10],
        index=1,
    )


    if st.button(
        "Generate quiz",
        key="generate_quiz",
    ):

        try:

            with st.spinner(
                "Generating questions from your PDF..."
            ):
                generated_questions = generate_quiz(
                    active_vector_store,
                    quiz_count,
                )
                st.session_state.quiz_questions = (
                    generated_questions
                )
                st.session_state.quiz_answers = {}
                st.session_state.quiz_checked = False
        except Exception as quiz_error:
            st.error(
                f"Unable to generate the quiz: {quiz_error}"
            )
    quiz_questions = st.session_state.quiz_questions
    if quiz_questions:
        with st.form("quiz_form"):

            selected_answers = {}


            for quiz_index, question_item in enumerate(
                quiz_questions
            ):

                st.markdown(
                    f"**Question {quiz_index + 1}: "
                    f"{question_item['question']}**"
                )
                selected_answers[str(quiz_index)] = st.radio(
                    "Choose your answer",
                    options=["A", "B", "C", "D"],
                    format_func=lambda option, opts=question_item["options"]: (
                        f"{option}. {opts[option]}"
                    ),
                    key=(
                        f"quiz_answer_"
                        f"{current_pdf_hash}_{quiz_index}"
                    ),

                    index=None,
                )

                st.divider()

            submitted = st.form_submit_button(
                "Check my answers"
            )

        if submitted:
            st.session_state.quiz_answers = selected_answers
            st.session_state.quiz_checked = True

        # Display quiz results.
        if st.session_state.quiz_checked:
            score = 0
            for quiz_index, question_item in enumerate(
                quiz_questions
            ):
                chosen = st.session_state.quiz_answers.get(
                    str(quiz_index)
                )
                correct_choice = question_item["correct_answer"]
                if chosen == correct_choice:
                    score += 1
                    st.success(
                        f"Question {quiz_index + 1}: Correct!"
                    )

                else:

                    st.error(
                        f"Question {quiz_index + 1}: "
                        f"The correct answer is {correct_choice}."
                    )
                st.write(
                    question_item.get("explanation", "")
                )
            st.subheader(
                f"Your score: {score} / {len(quiz_questions)}"
            )
            if st.button(
                "Try a new quiz",
                key="new_quiz",
            ):

                st.session_state.quiz_questions = []

                st.session_state.quiz_answers = {}

                st.session_state.quiz_checked = False

                st.rerun()

# ==========================================================
# 21. PDF ANALYTICS DASHBOARD
# FEATURE 7
# ==========================================================
with tab_analytics:

    st.subheader("Document Analytics")

    stats = st.session_state.pdf_stats


    if stats:

        # First row of metrics.
        col1, col2, col3 = st.columns(3)


        col1.metric(
            "PDF pages",
            stats.get("page_count", "Unknown"),
        )


        col2.metric(
            "Pages with readable text",
            stats.get("pages_with_text", "Unknown"),
        )

        col3.metric(
            "Search chunks",
            stats.get("chunk_count", "Unknown"),
        )


        # Second row of metrics.
        col4, col5 = st.columns(2)


        col4.metric(
            "Extracted characters",
            f"{stats.get('total_characters', 0):,}",
        )

        col5.metric(
            "Processing time",
            f"{stats.get('processing_time', 'Unknown')} seconds",
        )


        # OCR results.
        ocr_pages = stats.get(
            "ocr_pages",
            [],
        )


        st.markdown("### OCR results")


        if ocr_pages:

            st.write(
                "OCR was attempted on PDF pages: "
                + ", ".join(map(str, ocr_pages))
            )

        else:

            st.write(
                "No pages required OCR during the latest index build."
            )


        st.caption(
            "Analytics describe extracted text and indexed chunks. "
            "OCR attempts do not guarantee that every scanned page "
            "was recognized correctly."
        )


        # Chat statistics.
        st.markdown("### Chat statistics")

        user_questions = sum(
            1
            for history_item in st.session_state.chat_history
            if history_item["role"] == "user"
        )

        assistant_answers = sum(
            1
            for history_item in st.session_state.chat_history
            if history_item["role"] == "assistant"
        )

        col6, col7 = st.columns(2)

        col6.metric(
            "Questions asked",
            user_questions,
        )

        col7.metric(
            "Answers generated",
            assistant_answers,
        )

    else:
        st.info(
            "No analytics are available for this PDF yet."
        )