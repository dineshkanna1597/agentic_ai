# Agentic AI Job Board

An agentic AI-powered recruitment knowledge platform that processes
candidate resumes and job postings, validates and evaluates documents,
generates vector embeddings, and enables natural-language semantic
search through an Agentic RAG workflow.

Built with **Python, LangChain, LangGraph, LiteLLM, MCP, Qdrant, Ollama,
Celery, and Redis**.

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Key Features](#key-features)
-   [Architecture](#architecture)
-   [Document Ingestion Workflow](#document-ingestion-workflow)
-   [Agentic RAG Workflow](#agentic-rag-workflow)
-   [Agents and Responsibilities](#agents-and-responsibilities)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Prerequisites](#prerequisites)
-   [LLM and Embedding Models](#llm-and-embedding-models)
-   [Environment Configuration](#environment-configuration)
-   [Installation](#installation)
-   [Running the Application](#running-the-application)
-   [Document Processing](#document-processing)
-   [MCP Tools](#mcp-tools)
-   [Vector Database](#vector-database)
-   [Testing](#testing)
-   [Logging](#logging)
-   [Troubleshooting](#troubleshooting)
-   [Current Limitations](#current-limitations)
-   [Future Enhancements](#future-enhancements)

------------------------------------------------------------------------

## Overview

The Agentic AI Job Board is a backend-focused project that explores
agent-based document processing and retrieval for recruitment use cases.

It supports two primary workflows:

1.  **Document ingestion:** Monitors directories for resumes and job
    postings, validates incoming files, processes their content,
    evaluates document quality, generates embeddings, and stores vectors
    in Qdrant.
2.  **Agentic retrieval:** Accepts natural-language queries, selects
    appropriate MCP search tools, retrieves relevant resume or
    job-posting content, verifies requested skills, and generates a
    grounded response.

The ingestion and retrieval workflows are implemented as separate
LangGraph workflows. They share the LLM gateway, embedding
infrastructure, and Qdrant knowledge base.

### Primary use cases

-   Find candidates based on technical skills and experience.
-   Search job postings using natural-language descriptions.
-   Identify relevant resumes for a particular role.
-   Retrieve job postings matching a candidate's skill set.
-   Validate incoming resumes and job postings.
-   Evaluate document completeness and quality.
-   Build a searchable recruitment knowledge base.

------------------------------------------------------------------------

## Key Features

### Document ingestion

-   Asynchronous filesystem monitoring using Watchfiles.
-   Separate directories for resumes and job postings.
-   File extension and file-size validation.
-   File stability checks before processing.
-   Rejected-file quarantine.
-   Asynchronous task dispatch using Celery and Redis.
-   PDF, DOCX, and TXT text extraction.
-   Document text normalization.
-   Local PII detection and redaction using Presidio.
-   Resume contact extraction and validation.
-   LLM-assisted document classification and structured extraction.
-   Deterministic document validation.
-   LLM-assisted document quality assessment.
-   Deterministic quality checks and status assignment.
-   Agent-selected document chunking strategies.
-   Embedding generation and Qdrant storage.
-   Existing document vectors can be replaced during re-ingestion.

### Agentic retrieval

-   Natural-language candidate and job-posting search.
-   LangGraph-based agent orchestration.
-   MCP-based tool discovery and execution.
-   Semantic retrieval using Qdrant.
-   Separate resume and job-posting search tools.
-   Skill extraction and deterministic skill verification.
-   Document-level source collection and deduplication.
-   Configurable tool-call limit.
-   LLM fallback routing through LiteLLM.

### Infrastructure

-   Redis-backed Celery task queue.
-   Local Ollama model support.
-   Groq-hosted reasoning models.
-   LiteLLM model routing and fallbacks.
-   Persistent local Qdrant vector storage.
-   Structured application logging.
-   Rotating log files.
-   Pytest test suite.

------------------------------------------------------------------------

## Architecture

The application contains two LangGraph workflows:

-   `agentic_ingestion`: Processes, validates, and indexes documents.
-   `agentic_rag`: Retrieves relevant information and answers user
    queries.

### High-level architecture

``` mermaid
flowchart TB
    subgraph INGESTION["Document Ingestion"]
        A["Resume Directory"]
        B["Job Posting Directory"]
        C["Async File Watcher"]
        D["File Event Handler"]
        E["File Validation"]
        F["Celery Dispatcher"]
        G["Redis Queue"]
        H["Celery Worker"]
        I["Data Ingestion LangGraph"]
        J["Validation and Quality Assessment"]
        K["Chunking Agent"]
        L["Embedding Generation"]
        M["Qdrant Vector Store"]

        A --> C
        B --> C
        C --> D
        D --> E
        E --> F
        F --> G
        G --> H
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
    end

    subgraph RETRIEVAL["Agentic Retrieval"]
        N["User Query"]
        O["Agentic RAG LangGraph"]
        P["LLM Tool Selection"]
        Q["MCP Client"]
        R["MCP Server"]
        S["Resume Search Tool"]
        T["Job Posting Search Tool"]
        U["Qdrant Retrieval"]
        V["Skill Verification"]
        W["Grounded Response"]

        N --> O
        O --> P
        P --> Q
        Q --> R
        R --> S
        R --> T
        S --> U
        T --> U
        U --> V
        V --> W
    end

    M -.-> U

    subgraph MODELS["Model Infrastructure"]
        X["LiteLLM Router"]
        Y["Groq Reasoning Models"]
        Z["Ollama Local Models"]
        AA["Qwen3 Embedding Model"]

        X --> Y
        X --> Z
        AA --> L
        AA --> U
    end

    I -.-> X
    O -.-> X
```

### Document Ingestion Workflow

``` mermaid
flowchart TD
    A["Resume or Job Posting Added"] --> B["Watchfiles Event Detection"]
    B --> C["File Event Handler"]
    C --> D["File Validation"]
    D --> E{"Valid File?"}
    E -->|No| F["Move to Rejected Directory"]
    E -->|Yes| G["Dispatch Celery Task"]
    G --> H["Redis Queue"]
    H --> I["Celery Worker"]
    I --> J["Extract Document Text"]
    J --> K["Normalize Content"]
    K --> L["Detect and Redact PII"]
    L --> M["Extract Resume Contacts"]
    M --> N["Document Validation Agent"]
    N --> O{"Validation Accepted?"}
    O -->|No| R["Finalize Non-Accepted Document"]
    O -->|Yes| P["Document Quality Assessment"]
    P --> Q{"Quality Accepted?"}
    Q -->|No| R
    Q -->|Yes| S["Data Ingestion Agent"]
    S --> T["Select Chunking Strategy"]
    T --> U["Generate Embeddings"]
    U --> V["Store Vectors in Qdrant"]
    V --> W["Ingestion Completed"]
```

### Agentic RAG Workflow

``` mermaid
flowchart TD
    A["User Query"] --> B["Agentic RAG Agent"]
    B --> C["Understand Query"]
    C --> D["Select MCP Search Tool"]
    D --> E{"Search Type"}
    E -->|Candidate Search| F["search_resumes"]
    E -->|Job Search| G["search_job_postings"]
    F --> H["Generate Query Embedding"]
    G --> H
    H --> I["Qdrant Semantic Search"]
    I --> J["Retrieve Relevant Chunks"]
    J --> K["Collect and Deduplicate Sources"]
    K --> L["Extract Requested Skills"]
    L --> M["Verify Skills Against Retrieved Documents"]
    M --> N["Generate Grounded Response"]
```

------------------------------------------------------------------------

## Agents and Responsibilities

  -----------------------------------------------------------------------
  Agent                               Responsibility
  ----------------------------------- -----------------------------------
  `DataIngestionAgent`                Selects an appropriate chunking
                                      strategy and invokes the document
                                      chunking tool.

  `DocumentValidationAgent`           Assists with document
                                      classification and structured
                                      validation information.

  `DocumentQualityAgent`              Assesses document quality and
                                      provides structured quality
                                      results.

  `AgenticRAGAgent`                   Understands user queries, selects
                                      MCP tools, retrieves information,
                                      and generates responses.
  -----------------------------------------------------------------------

The ingestion graph combines agent decisions with deterministic
validation services. This helps keep document acceptance and quality
decisions from relying exclusively on LLM output.

------------------------------------------------------------------------

## Technology Stack

  Category                             Technology
  ------------------------------------ ------------------------------
  Language                             Python 3.12+
  Package and environment management   uv
  Agent framework                      LangChain
  Workflow orchestration               LangGraph
  LLM routing                          LiteLLM
  Hosted LLM provider                  Groq
  Local model runtime                  Ollama
  Embedding model                      Qwen3 Embedding 8B
  Tool communication                   Model Context Protocol (MCP)
  Task queue                           Celery
  Message broker and result backend    Redis
  Vector database                      Qdrant
  Filesystem monitoring                Watchfiles
  PII detection and anonymization      Microsoft Presidio
  PDF extraction                       PyMuPDF
  DOCX extraction                      docx2txt
  Configuration                        Pydantic Settings
  Testing                              Pytest and pytest-asyncio

------------------------------------------------------------------------

## Project Structure

The following is a simplified view of the source tree. Runtime data,
generated vector storage, logs, caches, and compiled files are excluded.

``` text
agentic_ai_job_board/
├── langgraph.json
├── pyproject.toml
├── requirements.txt
├── .env
├── .gitignore
├── notebook/
│   └── ingestion.ipynb
├── scripts/
│   └── run_watcher.py
└── src/
    └── agentic_ai_job_board/
        ├── agents/
        │   ├── agentic_rag/
        │   │   ├── agent.py
        │   │   ├── context.py
        │   │   ├── graph.py
        │   │   ├── prompts.py
        │   │   └── state.py
        │   ├── data_ingestion/
        │   │   ├── agent.py
        │   │   ├── context.py
        │   │   ├── graph.py
        │   │   ├── nodes.py
        │   │   ├── prompts.py
        │   │   ├── schemas.py
        │   │   └── state.py
        │   ├── document_quality/
        │   └── document_validation/
        ├── core/
        │   ├── celery_app.py
        │   ├── config.py
        │   └── logging.py
        ├── gateway/
        │   └── router.py
        ├── ingestion/
        │   ├── dispatcher.py
        │   ├── event_handler.py
        │   ├── events.py
        │   ├── extractor.py
        │   ├── file_validator.py
        │   ├── normalizer.py
        │   ├── quarantine.py
        │   └── watcher.py
        ├── mcp/
        │   ├── client.py
        │   ├── server.py
        │   └── tools/
        ├── rag/
        ├── services/
        │   ├── chunking/
        │   ├── embedding/
        │   ├── retrieval/
        │   ├── validation/
        │   ├── vector_store/
        │   └── verification/
        ├── tools/
        │   ├── chunking_tools.py
        │   ├── embedding_tools.py
        │   └── vector_store_tools.py
        ├── workers/
        │   └── ingestion_worker.py
        └── tests/
            ├── agents/
            ├── ingestion/
            ├── services/
            └── tools/
```

------------------------------------------------------------------------

## Prerequisites

Install the following before running the project:

-   Python 3.12 or newer.
-   [uv](https://docs.astral.sh/uv/)
-   [Redis](https://redis.io/)
-   [Ollama](https://ollama.com/)
-   A Groq API key.

The project uses local Qdrant persistent storage, so a separate Qdrant
server is not required for the current configuration.

------------------------------------------------------------------------

## LLM and Embedding Models

### Reasoning models

The LiteLLM router is configured with the following model chain:

  Priority     Provider   Model
  ------------ ---------- ----------------------------
  Primary      Groq       `groq/openai/gpt-oss-120b`
  Fallback 1   Groq       `groq/openai/gpt-oss-20b`
  Fallback 2   Ollama     `ollama/gpt-oss-20b`

The router uses retries and fallback configuration to improve
availability when a model request fails.

The local fallback requires the configured Ollama model to be available
locally. Verify that its model identifier matches the model tag
installed in Ollama. If your installed model uses a different tag,
update the local model configuration in `gateway/router.py`.

### Embedding model

The embedding router uses:

``` text
ollama/qwen3-embedding:8b
```

Pull the embedding model:

``` bash
ollama pull qwen3-embedding:8b
```

The embedding configuration currently expects vectors with a dimension
of **4096**. The embedding model and Qdrant collection dimensions must
remain consistent.

------------------------------------------------------------------------

## Environment Configuration

Create a `.env` file in the repository root.

``` env
GROQ_API_KEY=your_groq_api_key
```

The gateway requires `GROQ_API_KEY` to initialize the configured Groq
models.

Application settings are managed through Pydantic Settings. Other
configuration values, such as application environment, logging level,
Redis connection, file-size limits, and Qdrant settings, are defined in
`core/config.py`.

Do not commit `.env` or expose API keys in source control.

------------------------------------------------------------------------

## Installation

Clone the repository and enter the project directory:

``` bash
git clone <your-repository-url>
cd agentic_ai_job_board
```

Install dependencies using uv:

``` bash
uv sync
```

Verify the Python environment:

``` bash
uv run python --version
```

Make sure the `.env` file is configured before starting the application.

------------------------------------------------------------------------

## Running the Application

The ingestion and retrieval workflows use several independent processes.
Run the required services in separate terminal sessions.

### 1. Start Redis

On systems using systemd:

``` bash
sudo systemctl start redis-server
```

Verify Redis:

``` bash
redis-cli ping
```

Expected response:

``` text
PONG
```

The Celery configuration uses:

-   Broker: `redis://localhost:6379/0`
-   Result backend: `redis://localhost:6379/1`

### 2. Start Ollama

Ensure Ollama is running and accessible at:

``` text
http://localhost:11434
```

Verify available models:

``` bash
ollama list
```

Ensure the Qwen3 embedding model is installed and the configured local
reasoning fallback is available if you intend to use it.

### 3. Start the MCP server

``` bash
uv run python -m agentic_ai_job_board.mcp.server
```

The MCP server exposes its Streamable HTTP endpoint at:

``` text
http://127.0.0.1:8001/mcp
```

### 4. Start the LangGraph development server

``` bash
uv run langgraph dev --port 2024
```

The LangGraph API is available at:

``` text
http://127.0.0.1:2024
```

The graph identifiers are:

  Graph ID              Description
  --------------------- ------------------------------------------
  `agentic_rag`         Agentic retrieval and question answering
  `agentic_ingestion`   Document ingestion and indexing

### 5. Start the Celery worker

``` bash
uv run celery -A agentic_ai_job_board.core.celery_app:celery_app worker --loglevel=INFO
```

The worker consumes document-processing tasks from Redis and executes
the ingestion workflow.

### 6. Start the filesystem watcher

``` bash
uv run python scripts/run_watcher.py
```

The watcher monitors the configured resume and job-posting directories
and dispatches eligible file events for asynchronous processing.

------------------------------------------------------------------------

## Document Processing

### Input directories

Place documents in the following directories:

``` text
src/agentic_ai_job_board/data/
├── resumes/
├── postings/
└── rejected/
    ├── resume/
    └── posting/
```

The watcher initializes the monitored directories when required.

### Supported formats

  Format           Extension
  ---------------- -----------
  PDF              `.pdf`
  Microsoft Word   `.docx`
  Plain text       `.txt`

Maximum configured file size: **20 MB**.

### Processing stages

1.  The watcher detects a filesystem event.
2.  The event handler validates the file.
3.  Eligible resume and posting events are dispatched to Celery.
4.  The Celery worker invokes the data ingestion graph.
5.  The document is extracted and normalized.
6.  PII is detected and redacted locally.
7.  Resume contact details are extracted and validated.
8.  The document validation agent evaluates the document.
9.  Deterministic validation rules determine its acceptance status.
10. The document quality agent and quality service assess document
    quality.
11. The data ingestion agent selects a chunking strategy.
12. The chunking tool creates document chunks.
13. The embedding service generates vector embeddings.
14. The vector store persists the document chunks in Qdrant.

Documents that fail validation or quality requirements are finalized as
non-accepted rather than proceeding to vector storage.

------------------------------------------------------------------------

## MCP Tools

The MCP server exposes search tools for the retrieval workflow.

  -----------------------------------------------------------------------
  Tool                                Purpose
  ----------------------------------- -----------------------------------
  `search_resumes`                    Finds semantically relevant resume
                                      chunks.

  `search_job_postings`               Finds semantically relevant
                                      job-posting chunks.
  -----------------------------------------------------------------------

Both tools support query-based semantic search and expose parameters for
controlling result count and similarity thresholds.

Example resume search:

> Find candidates with Python, Django, and FastAPI experience.

Example job-posting search:

> Find backend developer roles requiring Python and REST API
> development.

The Agentic RAG workflow determines which search tool is appropriate
based on the user's query.

------------------------------------------------------------------------

## Vector Database

The project uses Qdrant with local persistent storage.

  Setting               Configuration
  --------------------- -------------------------
  Storage               Local persistent Qdrant
  Collection            `career_portal`
  Vector dimension      `4096`
  Embedding model       `qwen3-embedding:8b`
  Document categories   Resume and job posting

The retrieval service applies document-type filters so that resume
searches and job-posting searches retrieve the appropriate content.

Stored vector payloads include document information such as:

-   Document ID.
-   Source filename.
-   Document type.
-   Chunk index.
-   Chunk content.
-   Associated metadata.

The vector store service supports document-level operations such as
upserting chunks, checking document existence, replacing document
vectors, and deleting vectors by document ID.

The local persistent storage directory is:

``` text
src/agentic_ai_job_board/rag/qdrant_local_storage/
```

This directory contains generated database data and should not be
treated as source code.

------------------------------------------------------------------------

## Testing

Run the complete test suite:

``` bash
uv run pytest src/agentic_ai_job_board/tests/ -v
```

Run ingestion tests separately:

``` bash
uv run pytest src/agentic_ai_job_board/tests/ingestion/ -v
```

Run tests for a specific component by passing its test directory or file
path.

The latest recorded full test run completed with:

``` text
127 passed, 10 warnings
```

This is the result of a previous test run, not a guarantee for future
code changes.

------------------------------------------------------------------------

## Logging

Application logs are written to:

``` text
src/agentic_ai_job_board/logs/app.log
```

The logging configuration includes:

-   Timestamp.
-   Log level.
-   Module name.
-   Function name.
-   Source line number.
-   Log message.

Log rotation is configured with a maximum file size of **10 MB** and up
to **5 backup files**.

------------------------------------------------------------------------

## Troubleshooting

### Missing Groq API key

If the application reports that `GROQ_API_KEY` is missing:

1.  Verify that `.env` exists in the repository root.
2.  Add a valid Groq API key.
3.  Restart the affected process.

### Redis connection failure

Verify Redis is running:

``` bash
redis-cli ping
```

If Redis is unavailable, start the Redis service and restart the Celery
worker and watcher.

### Ollama model not found

Check installed models:

``` bash
ollama list
```

Pull the embedding model if necessary:

``` bash
ollama pull qwen3-embedding:8b
```

For the local reasoning fallback, verify that the installed Ollama model
tag matches the identifier configured in `gateway/router.py`.

### MCP connection failure

Verify that the MCP server is running and listening on port `8001`.

The configured endpoint is:

``` text
http://127.0.0.1:8001/mcp
```

### File rejected during ingestion

Check the following:

-   The file extension is `.pdf`, `.docx`, or `.txt`.
-   The file size does not exceed 20 MB.
-   The file is not empty or malformed.
-   The file has finished being copied before processing begins.

Inspect the application logs for the validation or quarantine reason.

### Qdrant storage lock

Avoid running multiple processes that write to the same local Qdrant
persistent storage directory concurrently. Stop other processes using
the collection before restarting ingestion.

### Deleted source files

The current dispatcher skips deleted-file events. Deleting a resume or
job posting from its input directory does not automatically remove its
vectors from Qdrant. Vector cleanup must be handled separately.

------------------------------------------------------------------------

## Current Limitations

-   This is a backend-focused portfolio project; an integrated frontend
    is not included.
-   Authentication and user-account management are not currently
    implemented.
-   The Qdrant configuration uses local persistent storage.
-   Resume and job-posting files are processed from local directories.
-   The Agent Chat UI is a separate frontend and is not included in this
    repository.
-   Deleted source files do not automatically trigger vector deletion.
-   PII detection and redaction reduce exposure before LLM processing
    but cannot guarantee detection of every sensitive value.
-   Production deployment, distributed infrastructure, and production
    security hardening are outside the current scope.

------------------------------------------------------------------------

## Future Enhancements

Potential areas for further development include:

-   Integrate a dedicated frontend for candidate and job-posting search.
-   Add user accounts and authentication.
-   Implement automatic vector cleanup when source documents are
    deleted.
-   Add document-processing status tracking.
-   Improve retrieval evaluation and ranking.
-   Add more comprehensive agent observability and tracing.
-   Introduce production-ready distributed vector storage.
-   Add deployment automation and containerization.
-   Expand candidate-to-job matching and recommendation capabilities.

------------------------------------------------------------------------

## Project Status

This project is an evolving exploration of agentic AI architecture for
recruitment workflows. Its primary focus is asynchronous document
ingestion, multi-agent document evaluation, semantic retrieval, and
grounded question answering using LangGraph and MCP.
