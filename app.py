import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from google import genai
from google.genai import types

import chromadb
from pypdf import PdfReader


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Please create a .env file "
        "and add your Gemini API key."
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Maximum PDF size = 20 MB
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="traditional_handicraft_knowledge"
)


# =========================================================
# MODELS
# =========================================================

EMBEDDING_MODEL = "gemini-embedding-001"

GENERATION_MODEL = "gemini-3.5-flash"


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text,
    chunk_size=900,
    overlap=150
):
    """
    Split PDF text into smaller overlapping chunks.
    """

    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =========================================================
# GEMINI EMBEDDING
# =========================================================

def create_embedding(
    text,
    task_type
):
    """
    Create Gemini embedding for text.
    """

    response = client.models.embed_content(

        model=EMBEDDING_MODEL,

        contents=text,

        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )

    return response.embeddings[0].values


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(pdf_path):

    reader = PdfReader(
        str(pdf_path)
    )

    pages = []

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    full_text = "\n".join(pages).strip()

    if not full_text:

        raise ValueError(
            "The PDF does not contain readable text."
        )

    chunks = chunk_text(full_text)

    source_name = pdf_path.name

    added_chunks = 0

    for index, chunk in enumerate(chunks):

        embedding = create_embedding(
            chunk,
            "RETRIEVAL_DOCUMENT"
        )

        document_id = (
            f"{source_name}-{index}"
        )

        collection.upsert(

            ids=[document_id],

            embeddings=[embedding],

            documents=[chunk],

            metadatas=[
                {
                    "source": source_name,
                    "chunk": index
                }
            ]
        )

        added_chunks += 1

    return added_chunks


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PDF UPLOAD API
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    try:

        # Check file field
        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No PDF file was selected."
            }), 400


        file = request.files["file"]


        # Check filename
        if not file.filename:

            return jsonify({
                "success": False,
                "error": "Please select a PDF file."
            }), 400


        # Only PDF
        if not file.filename.lower().endswith(".pdf"):

            return jsonify({
                "success": False,
                "error": "Only PDF files are allowed."
            }), 400


        # Secure filename
        filename = secure_filename(
            file.filename
        )


        # Save PDF
        pdf_path = (
            UPLOAD_FOLDER / filename
        )

        file.save(pdf_path)


        # Process PDF
        chunk_count = process_pdf(
            pdf_path
        )


        return jsonify({

            "success": True,

            "message": (
                f"PDF uploaded successfully. "
                f"{chunk_count} knowledge chunks added."
            ),

            "source": filename,

            "chunks": chunk_count

        }), 200


    except Exception as e:

        print(
            "\nUPLOAD ERROR:"
        )

        print(
            str(e)
        )


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# =========================================================
# CHAT API
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    try:

        # Get JSON safely
        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "success": False,

                "answer": (
                    "Invalid request."
                )

            }), 400


        question = (
            data.get("message", "")
            .strip()
        )


        if not question:

            return jsonify({

                "success": False,

                "answer": (
                    "Please enter a question."
                )

            }), 400


        # Check knowledge base
        document_count = (
            collection.count()
        )


        if document_count == 0:

            return jsonify({

                "success": True,

                "answer": (
                    "Please upload a "
                    "traditional handicraft PDF first."
                ),

                "sources": []

            }), 200


        # =================================================
        # CREATE QUERY EMBEDDING
        # =================================================

        query_embedding = create_embedding(

            question,

            "RETRIEVAL_QUERY"

        )


        # =================================================
        # SEARCH CHROMADB
        # =================================================

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=min(
                5,
                document_count
            )
        )


        documents = (
            results.get(
                "documents",
                [[]]
            )[0]
        )


        metadatas = (
            results.get(
                "metadatas",
                [[]]
            )[0]
        )


        # No matching documents
        if not documents:

            return jsonify({

                "success": True,

                "answer": (
                    "Sorry, I don't have "
                    "that information in my "
                    "knowledge base."
                ),

                "sources": []

            }), 200


        # =================================================
        # CREATE CONTEXT
        # =================================================

        context_parts = []


        for document, metadata in zip(
            documents,
            metadatas
        ):

            source = metadata.get(
                "source",
                "Uploaded PDF"
            )

            context_parts.append(

                f"[Source: {source}]\n"
                f"{document}"

            )


        context = "\n\n".join(
            context_parts
        )


        # =================================================
        # RAG PROMPT
        # =================================================

        prompt = f"""

You are the Traditional Handicraft
Knowledge Assistant.

Your job is to answer questions ONLY
using information from the uploaded
PDF knowledge base.

IMPORTANT RULES:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT invent information.
4. If the answer is not available in
   the context, reply exactly:

"Sorry, I don't have that information
in my knowledge base."

5. Keep answers simple and clear.
6. Do not answer unrelated questions.

========================
UPLOADED PDF CONTEXT
========================

{context}

========================
USER QUESTION
========================

{question}

========================
ANSWER
========================

"""


        # =================================================
        # GENERATE ANSWER
        # =================================================

        response = client.models.generate_content(

            model=GENERATION_MODEL,

            contents=prompt

        )


        answer = (
            response.text
            if response.text
            else
            "Sorry, I could not generate an answer."
        )


        # =================================================
        # SOURCES
        # =================================================

        sources = sorted(
            set(
                metadata.get(
                    "source",
                    "Uploaded PDF"
                )

                for metadata in metadatas

                if metadata
            )
        )


        return jsonify({

            "success": True,

            "answer": answer,

            "sources": sources

        }), 200


    except Exception as e:

        print(
            "\nCHAT ERROR:"
        )

        print(
            str(e)
        )


        return jsonify({

            "success": False,

            "error": str(e),

            "answer": (
                "Sorry, an error occurred "
                "while processing your question."
            )

        }), 500


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return jsonify({

        "success": False,

        "error": (
            "PDF file is too large. "
            "Maximum size is 20 MB."
        )

    }), 413


@app.errorhandler(
    404
)
def page_not_found(error):

    return jsonify({

        "success": False,

        "error": "Page not found."

    }), 404


@app.errorhandler(
    500
)
def internal_error(error):

    return jsonify({

        "success": False,

        "error": "Internal server error."

    }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "Traditional Handicraft RAG Chatbot"
    )

    print(
        "=========================================="
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print(
        "==========================================\n"
    )

    app.run(
        debug=True
    )
