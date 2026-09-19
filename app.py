import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types
import chromadb
from pypdf import PdfReader

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise RuntimeError('GEMINI_API_KEY is missing. Add it to your .env file.')
client = genai.Client(api_key=API_KEY)
app = Flask(__name__)
UPLOAD_FOLDER = Path('uploads'); UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
chroma_client = chromadb.PersistentClient(path='chroma_db')
collection = chroma_client.get_or_create_collection(name='traditional_handicraft_knowledge')
EMBED_MODEL = 'gemini-embedding-001'
GEN_MODEL = 'gemini-3.5-flash'

def chunk_text(text, chunk_size=900, overlap=150):
    text=' '.join(text.split()); chunks=[]; start=0
    while start < len(text):
        end=start+chunk_size; chunk=text[start:end].strip()
        if chunk: chunks.append(chunk)
        if end >= len(text): break
        start=end-overlap
    return chunks

def embed(text, task_type):
    r=client.models.embed_content(model=EMBED_MODEL, contents=text, config=types.EmbedContentConfig(task_type=task_type))
    return r.embeddings[0].values

def process_pdf(pdf_path):
    reader=PdfReader(str(pdf_path)); text='\n'.join((p.extract_text() or '') for p in reader.pages).strip()
    if not text: raise ValueError('No readable text was found in the PDF.')
    chunks=chunk_text(text); source=pdf_path.name
    for i, chunk in enumerate(chunks):
        collection.upsert(ids=[f'{source}-{i}'], embeddings=[embed(chunk,'RETRIEVAL_DOCUMENT')], documents=[chunk], metadatas=[{'source':source,'chunk':i}])
    return len(chunks)

@app.route('/')
def home(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files: return jsonify(error='Please select a PDF file.'),400
    file=request.files['file']
    if not file.filename: return jsonify(error='Please select a PDF file.'),400
    if not file.filename.lower().endswith('.pdf'): return jsonify(error='Only PDF files are supported.'),400
    path=UPLOAD_FOLDER/secure_filename(file.filename); file.save(path)
    try: return jsonify(message=f'PDF uploaded successfully. {process_pdf(path)} knowledge chunks added.', source=path.name)
    except Exception as e:
        if path.exists(): path.unlink()
        return jsonify(error=str(e)),500

@app.route('/chat', methods=['POST'])
def chat():
    q=((request.get_json(silent=True) or {}).get('message') or '').strip()
    if not q: return jsonify(answer='Please enter a question.'),400
    if collection.count()==0: return jsonify(answer='Please upload a handicraft PDF first.')
    try:
        qr=embed(q,'RETRIEVAL_QUERY'); results=collection.query(query_embeddings=[qr], n_results=min(5,collection.count()))
        docs=results.get('documents',[[]])[0]; metas=results.get('metadatas',[[]])[0]
        if not docs: return jsonify(answer="Sorry, I don't have that information in my knowledge base.")
        context='\n\n'.join(f"[Source: {m.get('source','PDF')}]\n{d}" for d,m in zip(docs,metas))
        prompt=f'''You are a Traditional Handicraft Knowledge Assistant.\nAnswer using ONLY the provided context. If unavailable, say: "Sorry, I don't have that information in my knowledge base." Do not invent facts.\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{q}'''
        r=client.models.generate_content(model=GEN_MODEL, contents=prompt)
        return jsonify(answer=r.text or 'Sorry, I could not generate an answer.', sources=sorted(set(m.get('source','PDF') for m in metas if m)))
    except Exception as e: return jsonify(error=str(e)),500

if __name__=='__main__': app.run(debug=True)
