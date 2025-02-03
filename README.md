## RAG System Setup

1. Clone repository:
```bash
git clone https://github.com/Vearance/rag-system-docker.git
cd docker
```
2. Ollama install:
```bash
docker pull ollama/ollama:latest
```

3. To use GPU, uncomment lines in compose.yaml

4. Start services:
```bash
docker compose up --build
```
5. Try: -> change as yours
```bash
curl -X POST "http://localhost:8900/documents/upload" `
  -F "file=@{file_name}"
```