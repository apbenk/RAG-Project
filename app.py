import os, re, json, fitz, faiss, numpy as np, mysql.connector
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from contextlib import contextmanager

app = Flask(__name__)
app.secret_key = "234h9ruewifj"
CORS(app)

# --- KONFIGURASI ---
DB_CONFIG = {"host": "localhost", "user": "root", "password": "", "database": "db_rag"}
DIRS = {k: os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", k) 
        for k in ["pdf", "text", "chunks", "faiss"]}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
ai_client = OpenAI(api_key="AIzaSyB6V3...", 
                   base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# --- HELPER ---
@contextmanager
def db_cursor(commit=False):
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)
    try:
        yield cur
        if commit: conn.commit()
    finally:
        cur.close()
        conn.close()

def process_pdf(file, user_id):
    """Menangani logika upload, ekstraksi, embedding, dan simpan."""
    fname = file.filename
    pdf_path = os.path.join(DIRS["pdf"], fname)
    file.save(pdf_path)

    # 1. Ekstrak Teks
    with fitz.open(pdf_path) as doc:
        text = " ".join([p.get_text("text") for p in doc])
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hitung panjang karakter
    char_length = len(text)
    
    # 2. Chunking
    CHUNK_SIZE = 200
    OVERLAP = 50 
    STEP = CHUNK_SIZE - OVERLAP

    if len(text) < CHUNK_SIZE:
        chunks = [text]
    else:
        chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), STEP)]
    
    # 3. Embedding
    embeddings = embedding_model.encode(chunks)
    
    # 4. Simpan File Fisik
    with open(os.path.join(DIRS["text"], f"{fname}.txt"), "w", encoding="utf-8") as f: f.write(text)
    
    with open(os.path.join(DIRS["chunks"], f"{fname}.txt"), "w", encoding="utf-8") as f:
        f.write("\n---CHUNK_SEP---\n".join(chunks))
    
    # 5. Simpan FAISS
    faiss_path = os.path.join(DIRS["faiss"], f"{fname}.index")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, faiss_path)

    # 6. Simpan DB (Update: Tambah char_length dan chunk_size)
    with db_cursor(commit=True) as cur:
        cur.execute("""
            INSERT INTO documents (filename, pdf_path, faiss_path, user_id, char_length, chunk_size) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """, (fname, pdf_path, faiss_path, user_id, char_length, CHUNK_SIZE))
    
    return len(chunks), len(text)

# --- ROUTES ---
@app.route("/")
def home(): return render_template("index.html", username=session.get('username')) if 'user_id' in session else redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET": return render_template("login.html")
    data = request.json
    with db_cursor() as cur:
        cur.execute("SELECT * FROM users WHERE username = %s", (data.get("username"),))
        user = cur.fetchone()
    
    if user and check_password_hash(user['password'], data.get("password")):
        session.update({'user_id': user['id'], 'username': user['username']})
        return jsonify({"message": "Login berhasil"})
    return jsonify({"error": "Login gagal"}), 401

@app.route("/logout")
def logout(): session.clear(); return redirect("/login")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET": return render_template("register.html")
    d = request.json
    try:
        with db_cursor(commit=True) as cur:
            cur.execute("INSERT INTO users (nama, email, username, password) VALUES (%s, %s, %s, %s)", 
                        (d.get("nama"), d.get("email"), d.get("username"), generate_password_hash(d.get("password"))))
        return jsonify({"message": "Berhasil"})
    except: return jsonify({"error": "Username exist"}), 400

@app.route("/documents")
def list_documents():
    if 'user_id' not in session: return jsonify([]), 401
    with db_cursor() as cur:
        cur.execute("SELECT id, filename FROM documents WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
        return jsonify(cur.fetchall())

@app.route("/upload", methods=["POST"])
def upload():
    if 'user_id' not in session: return jsonify({"error": "Auth required"}), 401
    try:
        chunks, chars = process_pdf(request.files['file'], session['user_id'])
        return jsonify({"message": "Sukses", "jumlChunks": chunks, "jmlChar": chars})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    if 'user_id' not in session: return jsonify({"error": "Auth required"}), 401
    try:
        q, doc_ids = request.form.get("question"), json.loads(request.form.get("document_ids", "[]"))
        if not doc_ids: return jsonify({"error": "Pilih dokumen"}), 400

        # Ambil path dokumen beserta metadata (char_length, chunk_size)
        with db_cursor() as cur:
            format_strings = ','.join(['%s'] * len(doc_ids))
            # Update select query untuk mengambil metadata baru
            cur.execute(f"SELECT filename, faiss_path, char_length, chunk_size FROM documents WHERE id IN ({format_strings})", tuple(doc_ids))
            docs = cur.fetchall()

        # Search Logic
        q_vec = np.array(embedding_model.encode([q]))
        candidates = []
        
        for doc in docs:
            try:
                index = faiss.read_index(doc['faiss_path'])
                with open(os.path.join(DIRS["chunks"], f"{doc['filename']}.txt"), "r", encoding="utf-8") as f:
                    chunks = f.read().split("\n---CHUNK_SEP---\n")
                
                D, I = index.search(q_vec, k=3)
                for score, idx in zip(D[0], I[0]):
                    if idx < len(chunks):
                        # Tambahkan metadata doc ke candidate
                        candidates.append({
                            "score": float(score), 
                            "text": chunks[idx], 
                            "source": doc['filename'],
                            "char_length": doc.get('char_length', 0),
                            "chunk_size": doc.get('chunk_size', 500)
                        })
            except: continue

        candidates.sort(key=lambda x: x["score"])
        top_res = candidates[:5]
        
        if not top_res: return jsonify({"answer": "Data tidak ditemukan", "context": ""})

        # LOGGING KE DB (Simpan Top Match Terbaik)
        # Kita ambil match terbaik (index 0) untuk dicatat di log statistik
        best_match = top_res[0]
        with db_cursor(commit=True) as cur:
             cur.execute("""
                INSERT INTO search_logs (user_id, filename, chunk_size, char_length, l2_score, question)
                VALUES (%s, %s, %s, %s, %s, %s)
             """, (session['user_id'], best_match['source'], best_match['chunk_size'], 
                   best_match['char_length'], best_match['score'], q))

        context_str = "\n\n".join([f"[Score: {c['score']:.4f}] [Src: {c['source']}]\n{c['text']}" for c in top_res])
        prompt = f"Context:\n{context_str}\n\nQuestion: {q}\nAnswer based on context:"
        
        ans = ai_client.chat.completions.create(model="gemini-2.5-flash", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

        # Simpan History
        with db_cursor(commit=True) as cur:
            cur.execute("INSERT INTO qa_history (user_id, question, answer, context) VALUES (%s, %s, %s, %s)",
                        (session['user_id'], q, ans, context_str))
            
        return jsonify({"answer": ans, "context": context_str})
    except Exception as e: 
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def get_history():
    if 'user_id' not in session: return jsonify([]), 401
    with db_cursor() as cur:
        cur.execute("SELECT question, answer, context, created_at FROM qa_history WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
        return jsonify(cur.fetchall())

# Route Baru: Ambil Data Logs
@app.route("/logs")
def get_logs():
    if 'user_id' not in session: return jsonify([]), 401
    with db_cursor() as cur:
        cur.execute("""
            SELECT user_id, filename, chunk_size, char_length, l2_score, question, created_at 
            FROM search_logs 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (session['user_id'],))
        return jsonify(cur.fetchall())

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8000, debug=True)
