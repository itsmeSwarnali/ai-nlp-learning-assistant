# Multimodal AI/NLP Learning Assistant

A Streamlit-based learning assistant that answers AI and NLP questions using Retrieval-Augmented Generation (RAG), an indexed collection of local study notes, intent-aware prompting, conversation history, and optional image analysis.

> **Current scope:** The assistant retrieves information from pre-indexed `.txt` notes stored in `data/notes` and accepts JPG, JPEG, and PNG image uploads. It does **not currently support user-uploaded PDF, DOCX, or TXT documents**.

## Live Demo

[Open the deployed Streamlit application](https://ai-nlp-learning-assistant.streamlit.app/)

## Project Overview

Learning Artificial Intelligence and Natural Language Processing often requires switching between lecture notes, documentation, diagrams, code examples, and online explanations. This project brings those activities into one conversational interface.

The assistant can:

- Answer AI and NLP questions using semantically retrieved study-note context
- Explain concepts in beginner-friendly language
- Maintain conversation history for follow-up questions
- Detect the likely intent of a question
- Analyse uploaded diagrams, screenshots, lecture-note images, and code images
- Combine image analysis with retrieved textual context
- Answer in the same language used by the user, including English and Bangla

## Key Features

### Retrieval-Augmented Generation

The application loads local `.txt` study notes from `data/notes`, splits them into overlapping chunks, converts the chunks into embeddings, and stores them in a Chroma vector database.

For each question, the retriever selects the three most similar chunks and provides them as context to the language model.

### Semantic Search

The project uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model to represent note chunks and user questions as vectors for similarity-based retrieval.

### Intent Detection

A Transformer-based intent classifier detects the likely purpose of the user’s message. The current categories include:

- Concept explanation
- Code help
- Project guidance
- Career guidance
- General conversation

When the trained classifier is unavailable, the application uses a lightweight keyword-based fallback.

### Multimodal Image Analysis

Users can attach JPG, JPEG, or PNG images through the Streamlit chat interface.

The assistant can analyse:

- AI and NLP diagrams
- Architecture drawings
- Lecture-note screenshots
- Code screenshots
- Graphs and visual explanations
- General educational images

The image explanation is combined with the user’s question and relevant retrieved notes before the final answer is generated.

### Conversation Memory

The application stores the current session’s chat history and passes previous messages to the RAG chain, allowing follow-up questions to retain conversational context.

### Language-Aware Responses

The prompting layer instructs the assistant to answer in the same language as the question. English questions receive English answers, while Bangla questions receive Bangla answers.

### Optional LangSmith Tracing

LangSmith environment variables can be configured to trace and inspect RAG executions during development and evaluation.

## System Architecture

```text
                    ┌──────────────────────────┐
                    │      Streamlit UI        │
                    │ Text question + image(s) │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │     Intent Classifier    │
                    │ Transformer or fallback  │
                    └─────────────┬────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
┌─────────────▼────────────┐             ┌────────────▼─────────────┐
│      Text RAG Path       │             │    Image Analysis Path   │
│                          │             │                          │
│ Local AI/NLP .txt notes │             │ JPG / JPEG / PNG image   │
│          ↓               │             │            ↓             │
│ Recursive text splitting│             │ Groq vision-capable LLM  │
│          ↓               │             │            ↓             │
│ Hugging Face embeddings │             │ Text image explanation   │
│          ↓               │             └────────────┬─────────────┘
│ Chroma vector database  │                          │
│          ↓               │                          │
│ Top-3 similar chunks    │                          │
└─────────────┬────────────┘                          │
              └───────────────────┬───────────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │ Prompt + Chat History    │
                    │ Intent + Retrieved Notes │
                    │ Optional Image Analysis  │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │      Groq-hosted LLM     │
                    └─────────────┬────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │     Final Explanation    │
                    └──────────────────────────┘
```

## Technology Stack

| Area | Technology |
|---|---|
| User interface | Streamlit |
| RAG framework | LangChain |
| Vector database | ChromaDB |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Text generation | Groq API with `llama-3.3-70b-versatile` |
| Image analysis | Groq API with `qwen/qwen3.6-27b` |
| Intent classification | Hugging Face Transformers and PyTorch |
| Conversation memory | LangChain message objects and Streamlit session state |
| Observability | LangSmith |
| Language | Python |

## How It Works

1. The application loads `.txt` files from `data/notes`.
2. Each document is divided into chunks of 700 characters with 100-character overlap.
3. The chunks are embedded using `all-MiniLM-L6-v2`.
4. ChromaDB stores the embeddings locally.
5. The user submits a question and may optionally attach an image.
6. The intent classifier predicts the question type.
7. The retriever selects the three most relevant note chunks.
8. Uploaded images are analysed by a vision-capable model.
9. Retrieved notes, image analysis, intent, question, and chat history are combined in the final prompt.
10. The language model generates a beginner-friendly response.

## Repository Structure

```text
AI-NLP-learning-Assistant-Chatbot_RAG/
├── app.py
├── requirements.txt
├── data/
│   └── notes/
│       └── *.txt
├── models/
│   └── intent_classifier/
├── src/
│   ├── ingestion.py
│   ├── vectorstore.py
│   ├── rag_chain.py
│   ├── intent_classifier.py
│   └── image_analyzer.py
├── vectorstore/
│   └── chroma_db/
├── .env
└── README.md
```

> Some generated or local directories may be excluded from Git through `.gitignore`.

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/itsmeSwarnali/AI-NLP-learning-Assistant-Chatbot_RAG.git
cd AI-NLP-learning-Assistant-Chatbot_RAG
```

### 2. Create a virtual environment

Using Conda:

```bash
conda create -n rag_env python=3.10 -y
conda activate rag_env
```

Or using `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Optional LangSmith configuration:

```env
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=AI-NLP-Learning-Assistant
```

Do not commit the `.env` file or API keys to GitHub.

### 5. Add or review the indexed notes

Place AI/NLP learning material as `.txt` files inside:

```text
data/notes/
```

The current ingestion pipeline reads `.txt` files from this folder when the application starts.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Questions

```text
Explain the difference between stemming and lemmatization.
```

```text
Why is attention useful in sequence models?
```

```text
Help me understand this Python traceback.
```

```text
How should I structure an NLP classification project?
```

You can also upload an AI/NLP diagram, code screenshot, or lecture-note image and ask:

```text
Explain this image step by step.
```

## Current Scope and Limitations

- The text knowledge base is created from local `.txt` files in `data/notes`.
- Users cannot currently upload PDF, DOCX, or TXT documents for temporary indexing.
- File attachments are limited to JPG, JPEG, and PNG images.
- Retrieval currently uses similarity search with the top three chunks.
- The final response does not yet display structured source citations.
- Retrieval quality has not yet been measured with a formal evaluation dataset.
- The application depends on external Groq models and API availability.
- Model names may need to be updated if the provider changes its available models.
- Chat history exists only for the active Streamlit session.
- Generated answers may contain mistakes and should be verified for important academic or professional use.

## Planned Improvements

- [ ] Add PDF, DOCX, and TXT upload
- [ ] Create a temporary per-user vector store for uploaded documents
- [ ] Display source names and retrieved passages with each answer
- [ ] Add retrieval evaluation metrics such as Recall@K and MRR
- [ ] Add a small question-answer evaluation set
- [ ] Resolve and standardise all intent labels
- [ ] Add clearer user-facing exception handling
- [ ] Add API-rate and token-usage protection
- [ ] Add automated tests
- [ ] Add GitHub Actions for testing and code quality
- [ ] Pin and separate production dependencies
- [ ] Add feedback controls for answer quality

## Responsible Use

This application is an educational assistant, not an authoritative academic source. Users should verify generated explanations, code, and career guidance before relying on them in examinations, publications, production systems, or professional decisions.

## Author

**Swarnali Mollick**

- GitHub: [itsmeSwarnali](https://github.com/itsmeSwarnali)
- Portfolio: [swarnalimollick.vercel.app](https://swarnalimollick.vercel.app)
- LinkedIn: [s-mollick](https://www.linkedin.com/in/s-mollick/)
- Google Scholar: [Swarnali Mollick](https://scholar.google.com/citations?user=1savjdEAAAAJ&hl=en)

## License

No license is currently specified in this repository. Add a license before inviting reuse or external contributions.
