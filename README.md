## RAG System Setup

1. Clone repository:
```bash
git clone https://github.com/Vearance/rag-system-docker.git
```
2. Start services:
```bash
docker compose up --build
```
3. Try: -> change as yours
```bash
curl -X POST "http://localhost:8900/documents/upload" `
  -F "file=@{file_name}"
```