# 🎓 EduMentor-OS

> An open-source, multi-agent AI system for automated grading, personalized tutoring, and learning analytics — powered by **LangGraph** and **LangChain**.

---

## 🏗️ Architecture Overview

EduMentor-OS is orchestrated by a **LangGraph StateGraph** that routes tasks to three specialized agents:

```
                    ┌──────────────────────────────┐
                    │        FastAPI Layer          │
                    │  POST /grade  POST /tutor     │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │     LangGraph Main Graph      │
                    │   (graphs/main_graph.py)      │
                    └──────┬───────────────┬────────┘
                           │               │
              ┌────────────▼───┐    ┌──────▼──────────┐
              │  GraderAgent   │    │   TutorAgent     │
              │ (PDF → Score)  │───►│ (Adaptive hints) │
              └────────────────┘    └──────┬───────────┘
                                           │
                                   ┌───────▼───────────┐
                                   │  AnalystAgent      │
                                   │ (Reports/Profiles) │
                                   └───────────────────┘
```

All agents share a **RAG vector store** (ChromaDB) backed by the `data/knowledge_base/` corpus.

---

## 📁 Project Structure

```
edumentor-os/
├── agents/              # LangGraph agent nodes
│   ├── grader_agent.py  # Automated grading with rubric evaluation
│   ├── tutor_agent.py   # Personalized tutoring & Q&A
│   └── analyst_agent.py # Learning analytics & reporting
├── tools/               # LangChain tools bound to agents
│   ├── pdf_parser.py    # Extract text from student PDF submissions
│   ├── rubric_retriever.py  # Load & search grading rubrics
│   ├── student_profile.py   # Read/write persistent student profiles
│   ├── code_executor.py     # Safe sandboxed code execution
│   └── quiz_generator.py    # Adaptive quiz generation
├── graphs/
│   └── main_graph.py    # LangGraph StateGraph orchestration
├── rag/
│   ├── vector_store.py  # ChromaDB ingestion & retrieval
│   └── embeddings.py    # Embedding model factory
├── api/
│   └── main.py          # FastAPI HTTP endpoints
├── data/
│   ├── sample_copies/   # Example student submissions (PDF)
│   ├── rubrics/         # Grading rubric definitions (JSON/YAML)
│   └── knowledge_base/  # Educational content for RAG ingestion
├── config.py            # Centralized settings (Pydantic BaseSettings)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-org/edumentor-os.git
cd edumentor-os
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Ingest the knowledge base

```bash
# Add documents (PDF, Markdown, .txt) to data/knowledge_base/
# Then trigger ingestion:
curl -X POST http://localhost:8000/ingest -H "X-Admin-Key: your_admin_key"
```

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## 🔑 Required API Keys

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | LLM inference + embeddings |
| `TAVILY_API_KEY` | Web search augmentation for AnalystAgent |
| `LANGSMITH_API_KEY` | Observability & tracing (optional but recommended) |
| `LANGSMITH_PROJECT` | LangSmith project name for trace grouping |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/grade` | Grade a student submission (PDF upload + rubric_id) |
| `POST` | `/tutor` | Start/continue a tutoring session |
| `GET`  | `/report/{class_id}` | Get analytics report for a class |
| `POST` | `/ingest` | Rebuild the RAG knowledge base index |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2+ |
| LLM / Embeddings | LangChain + OpenAI |
| Vector Store | ChromaDB |
| API | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| Web Search | Tavily |
| Config | Pydantic + python-dotenv |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
