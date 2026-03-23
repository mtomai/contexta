from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-5.2-chat-latest"
    light_model: str = "gpt-4o-mini"
    long_context_model: str = "gpt-4.1"
    vision_model: str = "gpt-4o-mini"

    # Storage Configuration
    chroma_db_path: str = "./chroma_db"
    uploads_path: str = "./uploads"

    # Document Processing Configuration
    max_file_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Parent-Child Chunking Configuration
    parent_chunk_size: int = 1500      # Tokens for parent chunks (broad context for LLM)
    child_chunk_size: int = 300        # Tokens for child chunks (precise search targets)
    child_chunk_overlap: int = 50      # Overlap between child chunks
    enable_parent_child: bool = True   # Enable parent-child retrieval strategy

    # Layout-Aware Parsing Configuration
    enable_layout_aware_parsing: bool = True  # Enable structural PDF/DOCX parsing
    header_font_ratio: float = 1.2            # Min font size ratio vs body to classify as header

    # CORS Configuration
    allowed_origins: list[str] = ["http://localhost:5172", "http://127.0.0.1:5172", "http://localhost:3000"]

    # RAG Pipeline Configuration
    default_top_k: int = 8
    child_retrieval_top_k: int = 15    # Child chunks to retrieve before parent mapping
    max_context_tokens: int = 12000    # Increased for parent-child context
    max_continuations: int = 5

    # Hybrid Search Configuration (BM25 + Vector)
    enable_hybrid_search: bool = True   # Enable BM25 + Vector hybrid search
    bm25_weight: float = 0.3           # Weight for BM25 in RRF fusion
    vector_weight: float = 0.7         # Weight for vector search in RRF fusion
    rrf_k: int = 60                    # RRF constant (higher = more uniform blending)

    # Embedding Configuration
    max_embedding_tokens: int = 8000
    embedding_cache_size: int = 1000

    # Query Router Thresholds
    high_relevance_threshold: float = 0.85
    low_relevance_threshold: float = 0.35

    # LLM Parameters
    compression_temperature: float = 0.0
    title_temperature: float = 1
    title_max_tokens: int = 20

    # --- LLM Prompts ---

    # Chat system prompt (used in chat.py for RAG responses)
    # --- LLM Prompts ---

    # Chat system prompt (used in chat.py for RAG responses)
    chat_system_prompt: str = (
        "You are an expert research analyst. Your task is to synthesize and answer user queries "
        "BASED EXCLUSIVELY on the provided document context.\n\n"
        "STRICT DIRECTIVES:\n"
        "1. ABSOLUTE GROUNDING: Your answers must derive solely from the provided text. "
        "Never use prior knowledge to add facts, dates, or concepts not present in the context.\n"
        "2. UNDERSTANDING CONTEXT: The context is provided in XML blocks formatted as "
        "<document name=\"...\"> and page=\"...\">. Read these attributes carefully.\n"
        "3. GRANULAR CITATIONS: Every single factual claim, data point, or key concept MUST be "
        "immediately followed by its exact source using the XML attributes. Use THIS EXACT FORMAT: "
        "[document_name, page X]. Do not use XML for your output citations. Do not group citations "
        "at the end; use inline citations directly after the claim.\n"
        "4. HANDLING MISSING DATA: If the context lacks the necessary information, state explicitly: "
        "'The provided documents do not contain information about...' "
        "Do not guess or hallucinate.\n"
        "5. NEUTRALITY: Maintain an objective tone. If sources conflict, present all versions neutrally with citations.\n"
        "6. FORMATTING: Use bullet points, bold text for key terms, and short paragraphs for extreme readability.\n\n"
        "CRITICAL: ALWAYS respond in the same language as the user's query."
    )

    # Chat user prompt template (placeholders: {context_text}, {query})
    chat_user_prompt_template: str = (
        "Below are the extracted snippets from the uploaded documents, formatted as XML blocks:\n\n"
        "{context_text}\n\n"
        "User Query: {query}\n\n"
        "Provide a comprehensive, highly structured response based strictly on the XML context above. "
        "Use the 'name' and 'page' attributes from the XML tags to append inline citations "
        "like [Document_Name, page X] immediately after every extracted fact. "
        "Respond in the same language as the user's query."
    )

    # Streaming system prompt (used in chat_streaming.py)
    chat_streaming_system_prompt: str = chat_system_prompt

    # Streaming user prompt template (placeholders: {context_text}, {query})
    chat_streaming_user_prompt_template: str = chat_user_prompt_template

    # Refinement system prompt (used for document refinement conversations)
    refinement_system_prompt: str = (
        "You are an analytical editorial reviewer. Your task is to modify, expand, or refine "
        "a previously generated document or response, strictly following the user's instructions.\n\n"
        "RULES:\n"
        "1. EXCLUSIVELY use the provided XML document context to add any new information.\n"
        "2. Apply the requested modifications (e.g., 'expand', 'summarize') while maintaining strict grounding in actual facts.\n"
        "3. Accurately preserve existing citations. When integrating new data from the XML tags (<document name=\"...\" page=\"...\">), "
        "add new citations using the strict format [document_name, page X].\n"
        "4. Do not remove correct information present in the previous version unless explicitly requested.\n\n"
        "CRITICAL: ALWAYS respond in the same language as the user's query."
    )

    # Refinement user prompt with previous response (placeholders: {previous_response}, {context_text}, {query})
    refinement_user_prompt_template: str = (
        "Previously generated document:\n{previous_response}\n\n---\n\n"
        "Context from original documents (XML format):\n{context_text}\n\n---\n\n"
        "Modification request: {query}\n\n"
        "Apply the requested modifications to the previous document. "
        "Cite new sources using the format [Document_Name, page X] derived from the XML tags. Respond in the same language as the user's query."
    )

    # Refinement user prompt without previous response (placeholders: {context_text}, {query})
    refinement_user_prompt_no_history_template: str = (
        "Document context (XML format):\n{context_text}\n\n---\n\n"
        "Query: {query}\n\n"
        "Answer based strictly on the provided XML documents. "
        "Cite sources using the format [Document_Name, page X] derived from the XML attributes. Respond in the same language as the user's query."
    )

    # Title generation prompt (placeholder: {first_message})
    # (Remains unchanged, context from documents not used)
    title_generation_prompt: str = (
        "Generate a short title (max 5-7 words) in the same language as the user's query for a conversation "
        "starting with this user query.\n"
        "The title must be descriptive and capture the essence of the question.\n\n"
        "Examples:\n"
        "- \"How does the authentication system work?\" -> \"Authentication System\"\n"
        "- \"What are the installation requirements?\" -> \"Installation Requirements\"\n"
        "- \"Explain the backup process\" -> \"Backup Process\"\n"
        "- \"What is this document about?\" -> \"Document Content\"\n\n"
        "Query: {first_message}\n\n"
        "Respond ONLY with the title in the same language as the query, without quotes or trailing punctuation."
    )

    # Context compression prompt (placeholders: {query}, {full_context})
    compression_prompt_template: str = (
        "You are a surgical extraction filter for a document retrieval system.\n"
        "Extract ONLY the concepts, data, and information from the SOURCE TEXT that are objectively "
        "useful to answer the QUERY.\n\n"
        "CRITICAL RULES:\n"
        "1. Do not alter the original meaning.\n"
        "2. The SOURCE TEXT is formatted in XML tags (<document name=\"...\" page=\"...\">). "
        "You MUST output your compressed text wrapped in the EXACT SAME XML tags so the downstream system "
        "knows where the information came from.\n"
        "3. Ruthlessly discard all irrelevant background noise.\n"
        "4. If there is absolutely nothing useful in a specific XML block, omit that block entirely.\n\n"
        "QUERY: {query}\n\n"
        "SOURCE TEXT TO COMPRESS:\n{full_context}\n\n"
        "OUTPUT: Compressed text preserving the original <document> XML tags."
    )

    # Continuation prompt (used when LLM response is truncated)
    continuation_prompt: str = "Continue the response from exactly where you left off. Respond in the same language you were using."

    # Full-document system prompt suffix (appended when query_type == FULL_DOCUMENT)
    full_document_suffix: str = (
        "\n\nGLOBAL VISION MODE: Attention, unlike usual, you are not reading isolated snippets "
        "but have been provided with the ENTIRE content of the documents formatted in XML. "
        "Leverage this global view to create cross-references between chapters, identify macro-themes, "
        "and provide a holistic synthesis. The obligation to insert granular citations [Document_Name, page X] remains mandatory."
    )

    # Fallback system prompt (used when chat_system_prompt is empty)
    fallback_system_prompt: str = (
        "You are an expert AI assistant in document analysis. "
        "Answer queries based exclusively on the provided XML context. "
        "If the context does not contain sufficient information, state so clearly. "
        "Always respond in the same language as the user's query."
    )

    # Vision image description prompt (used in document_parser.py)
    # (Remains unchanged)
    vision_prompt: str = (
        "Analyze this image, chart, or graph with extreme precision for a document retrieval system.\n"
        "1. If it's a Chart/Graph: extract the exact data points, identify the axes, labels, legends, and summarize the main trend.\n"
        "2. If it's a Diagram/Flowchart: explain the step-by-step workflow or relationships between elements.\n"
        "3. If it contains Text: extract the visible text accurately (OCR).\n"
        "Structure your response clearly using lists and key-value pairs so a text-based search engine "
        "can easily index and retrieve these specific details later. Respond in the primary language used inside the image."
    )

    # Multi-Query expansion prompt template (placeholder: {alternatives_needed})
    # (Remains unchanged)
    multi_query_system_prompt_template: str = (
        "You are an expert at reformulating search queries to maximise retrieval "
        "from a vector database.  Given a user question, generate exactly "
        "{alternatives_needed} alternative version(s) of the question.  "
        "Each alternative should use different wording, synonyms, or a slightly "
        "different angle while preserving the original intent.\n\n"
        "RULES:\n"
        "- Output ONLY the alternative queries, one per line.\n"
        "- Do NOT number them and do NOT use bullet points.\n"
        "- Keep the same language as the original query.\n"
        "- Do NOT repeat the original query."
    )

    # Agent executor: citation instruction appended to user prompt
    agent_citation_instruction: str = (
        "The context is provided in XML <document> tags. "
        "Cite sources using the attributes in the exact format [Document_Name, page X]."
    )

    # Agent executor: template output prefix format (placeholder: {template_content})
    agent_template_prefix: str = "\n\n---\nOutput template:\n{template_content}"

    # --- Fallback / error messages ---
    #In Italian because they are displayed directly on the UI to the user

    # No relevant results found (chat.py)
    no_results_message: str = (
        "I did not find relevant information in the uploaded documents "
        "to answer your question."
    )

    # Out-of-scope response (query_router.py)
    out_of_scope_message: str = (
        "I did not find relevant information about this topic "
        "in the uploaded documents. Try rephrasing the question or "
        "check if the documents contain the information you are looking for."
    )

    # No documents found (chat_streaming.py)
    no_documents_message: str = "No documents found. Please upload some documents first."

    # Agent executor: no content in selected documents
    agent_no_content_message: str = "I did not find content in the selected documents."

    # No refinement results (chat.py)
    no_refinement_results_message: str = (
        "I did not find relevant information in the documents for your request."
    )

    # --- Model detection ---

    # Prefixes that identify reasoning models (affects API call parameters)
    reasoning_model_prefixes: list[str] = ["o1", "o3", "o4"]

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()