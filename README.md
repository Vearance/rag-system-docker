# RAG System

## System Architecture Overview

### 📁 Folder Structure & Purpose

#### `app/`
- Main application directory.
- Contains subfolders like `data/vector_store`, `routes/`, and `src/`.

#### `data/vector_store/`
- Stores vector index data.
- **Files:**
  - `index.faiss` → FAISS index for efficient similarity search.
  - `index.pkl` → Pickle file, possibly storing metadata or additional vector information.

#### `routes/`
- Handles API endpoints.
- **Files:**
  - `documents.py` → API for managing document-related operations.
  - `questions.py` → API for handling question-based queries.

#### `src/`
- Contains core processing and logic modules.
- **Files:**
  - `config.py` → Application configurations.
  - `embeddings.py` → Generates text embeddings for vector storage.
  - `generation.py` → Possibly handles AI-generated responses.
  - `process.py` → Preprocessing logic for input data.
  - `retrieval.py` → Vector search and retrieval functions using FAISS.
  - `main.py` → Main entry point for the application.
  - `schemas.py` → Defines data models for API interactions.

#### Other Files & Directories:
- **`docker/`** → Possibly contains Docker configurations.
- **`.env`** → Environment variables (e.g., API keys, database credentials).
- **`.gitignore`** → Specifies files and directories to be ignored in Git.

### 🔍 Summary
This project appears to be an AI-powered vector search system, possibly for **Retrieval-Augmented Generation (RAG)** or **semantic search** using FAISS. The architecture is well-structured with API endpoints, vector storage, and core processing modules.
---

- [Go to Docker Setup Instruction](#docker-setup-instructions)

---

## Technology Stack Explanation

- **FastAPI**: A modern Python web framework for building APIs, based on standard Python type hints, that is fast and easy to use.
- **Docker**: Containerization technology for running the application and database in isolated environments.
- **Docker Compose**: A tool for defining and managing multi-container Docker applications.


### Libraries and Technologies Used:
- **FastAPI**: For building the API.
- **Docker**: To containerize the application and database.

---

## Docker Setup Instructions

### 1. Clone the Repository

First, clone the repository and navigate into the project directory by running the following command:

```bash
git clone https://github.com/Vearance/rag-system-docker.git
```

After Cloning the repository, navigate to the `Docker Folder Directory` by running the following command:

```bash
cd rag-system-docker/docker
```


### 2. Build the Docker containers

and build the Docker containers by running the following command:

```bash
docker-compose up --build
```

### 3. Start the Docker Containers

Once the containers are built, they should automatically start. If they're not running, you can manually start them with the following command:

```bash
docker-compose up
```

### 4. Install Model

Once the Docker container is running, execute the appropriate command based on your available RAM.

#### **For systems with < 9GB RAM Available:**
```bash
docker exec -it docker-ollama-1 ollama pull tinyllama
docker exec -it docker-ollama-1 ollama pull all-minilm
```

#### **For systems with > 9GB RAM Available:**
```bash
docker exec -it docker-ollama-1 ollama pull llama2
docker exec -it docker-ollama-1 ollama pull all-minilm
```

Note (for <9 GB RAM): After pulling llama2, update the MODEL_NAME in app/src/config.py to `tinyllama` to ensure the system uses the correct model.

### 5. Visit the Application

Now that your containers are running, you can access the FastAPI application by visiting:

```bash
http://localhost:8000
```