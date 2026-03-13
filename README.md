# Contexta

An Enterprise-grade RAG (Retrieval-Augmented Generation) application that allows you to upload PDF, Word, and Excel documents, organize them into notebooks, and interact with the content via chat.

Answers are based **exclusively** on uploaded documents, with surgical citations to sources. The architecture implements the most modern Information Retrieval techniques (Multi-Query, Hybrid Search, Re-Ranking) to zero out hallucinations and maximize precision. It also includes a system of **customizable Agents** to perform automatic and asynchronous analysis on documents.

## 🚀 Advanced RAG Architecture (Under the Hood)

This project is not limited to simple vector search, but implements a multi-stage pipeline to guarantee perfect results. Here's what happens behind the scenes:

### 1. Multi-Query Expansion (Pre-retrieval)

* **What it is:** Before searching, the LLM generates 2-3 variants of the user's question (using synonyms or reformulations).
* **Why:** Users often write short or imprecise queries (e.g., "2023 expenses"). If the document uses formal terms ("financial statement of the year"), classic vector search might fail. By generating variants, we maximize the probability (Recall) of intercepting the right document.

### 2. Parallel Hybrid Search (Vector + BM25 with Stemming)

* **What it is:** Queries are searched simultaneously on two engines: ChromaDB (Semantic/Vector Search) and BM25Okapi (Lexical/Keyword Search). BM25 has been enhanced with **Italian Stemmer (NLTK)** to reduce words to their root (e.g., "fatture" -> "fattur"). Searches happen in parallel via `asyncio.gather` so as not to block the server.
* **Why:** Vector search understands concepts but is terrible at searching for exact codes, proper names, or acronyms. BM25 compensates for this weakness. Italian stemming allows finding matches even if the word is declined differently between query and document.

### 3. Reciprocal Rank Fusion (RRF)

* **What it is:** A mathematical algorithm that merges the results of the Vector engine and the BM25 engine into a single ranking, assigning a balanced weight.
* **Why:** It allows having the best of both worlds, rewarding documents that respond well both semantically and by exact keywords.

### 4. Cross-Encoder Re-Ranking (Post-retrieval)

* **What it is:** The system initially retrieves many documents (e.g., 15-30). Before passing them to the LLM to generate the answer, a lightweight but very precise Cross-Encoder model re-reads the query and every single document, assigning a real relevance score and discarding those below a certain threshold.
* **Why:** Vector embedding only calculates "closeness" in space. The Cross-Encoder, instead, evaluates the actual logical correlation between Question and Answer, acting as a final filter against off-topic documents.

### 5. Layout-Aware Parent-Child Chunking

* **What it is:** During upload, the parser divides texts into small fragments (Child, ~300 tokens) for search, linking them to large fragments (Parent, ~1500 tokens) for context. Furthermore, the parser dynamically injects Markdown chapter titles inside the small Child Chunks (e.g., `[Section: Medical Expenses]`).
* **Why:** Prevents "Context Starvation". If a micro-fragment says only *"The limit is €500"*, ChromaDB would never know what it refers to. By injecting the title, the search engine understands the exact context of the phrase.

### 6. XML Prompting & Strict Citations

* **What it is:** The context passed to the final LLM is not simple text, but is formatted with structured XML tags (`<document name="File.pdf" page="3"> Text </document>`). In the System Prompt, the LLM is instructed to read the XML attributes to generate the citation in the format `[File.pdf, page 3]`.
* **Why:** Leverages the "Attention" mechanism of modern LLM models. Visually binding the XML attribute to the text zeros out hallucinations: the model will never invent a file name nor confuse pages.

---

## ✨ Key Features

### User Interface (UX)

* **Real-time Chain of Thought (AI Thought):** While the AI processes the pipeline (Multi-Query -> Search -> Re-Rank), the React interface shows a dynamic accordion with animated checks (✅) of the logical steps. The curtain closes automatically as soon as text generation begins via Server-Sent Events (SSE).
* **Interactive Source Visualization:** Every answer includes clickable citations. Clicking opens a panel showing the exact extract of the document used.

### Data Management and Performance

* **Strategic Embedding Caching:** Phrases searched by users are saved. If a query is repeated, the embedding is retrieved instantly without calling OpenAI APIs, saving time and costs.
* **Batch Database Fetching:** Access to the SQLite database to retrieve complete documents (Parent Chunks) is optimized with `IN (...)` queries, eliminating the "N+1" sequential query problem and slashing response times.
* **Document Upload:** Support for files up to 50MB (PDF, DOC/DOCX, XLSX). PDFs include OCR support via OpenAI Vision API to describe charts and images.

### Agent Executor

* **Asynchronous Execution:** Background agents can process and synthesize entire documents using `FULL_DOCUMENT` mode without blocking the main server.

---

## 🛠 Installation and Setup

1. **Clone the repository**
2. **Backend Setup (Python 3.10+) with uv**

   ```bash
   cd backend
   uv sync              # Create venv and install dependencies from pyproject.toml
   # To include dev dependencies (pytest, etc.) too:
   uv sync --extra dev
   # Install NLTK module for stemming (if required by code)
   uv run python -c "import nltk; nltk.download('punkt')"
   ```

3. **Environment Variables (Backend)**
Create a `.env` file in backend/:

   ```text
   OPENAI_API_KEY=your_api_key_here
   ALLOWED_ORIGINS=http://localhost:5172,http://localhost:3000
   ```

4. **Start the Backend**

   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```

5. **Frontend Setup (Node.js)**

   ```bash
   cd frontend
   npm install
   ```

6. **Environment Variables (Frontend)**
Create a `.env` file in frontend/:

   ```text
   VITE_API_URL=http://localhost:8000
   ```

7. **Start the Frontend**

   ```bash
   npm run dev
   ```

---

## 🐛 Troubleshooting

* Backend starts but search is slow: Check that you are using the asynchronous version of searches (asyncio.gather). The first start might require downloading the Cross-Encoder model (SentenceTransformers) into cache.

* Frontend does not connect: Check that the backend is running on port 8000 and that CORS (ALLOWED_ORIGINS) is configured correctly.

* Answers not accurate: Check that the uploaded file is not a scanned PDF of poor quality (the extracted text might be corrupted). The RAG pipeline does not invent data: if the text is missing in the file, the AI will answer that it did not find the information.
