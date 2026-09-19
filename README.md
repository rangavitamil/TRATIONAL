# Traditional Handicraft Knowledge Assistant

Flask + Gemini + ChromaDB RAG chatbot. Upload a PDF in the chatbot, retrieve relevant chunks, and answer from the uploaded knowledge base.

## Files
- app.py
- requirements.txt
- .gitignore
- .env.example
- README.md
- templates/index.html
- uploads/

## Environment
Create `.env` and add `GEMINI_API_KEY=your_actual_key`.
Do not upload `.env` to GitHub.

## RAG flow
PDF Upload -> Text Extraction -> Chunking -> Gemini Embedding -> ChromaDB -> Retrieval -> Gemini Answer
