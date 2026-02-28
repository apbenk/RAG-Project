import os
import re
import fitz  
import faiss
import numpy as np
import mysql.connector
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
import json # Tambahkan ini
# ==========================================
# 1. KONFIGURASI GLOBAL
# ==========================================

app = Flask(__name__)
app.secret_key = "234h9ruewifj"
CORS(app)

# Konfigurasi Database
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "db_rag"
}

# Konfigurasi Direktori Penyimpanan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE = {
    "pdf": os.path.join(BASE_DIR, "storage", "pdf"),
    "text": os.path.join(BASE_DIR, "storage", "text"),
    "chunks": os.path.join(BASE_DIR, "storage", "chunks"),
    "faiss": os.path.join(BASE_DIR, "storage", "faiss")
}

# Membuat folder jika belum ada
for path in STORAGE.values():
    os.makedirs(path, exist_ok=True)

# Inisialisasi Model embedding
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Client untuk Chat (Menggunakan Gemini via library OpenAI)
ai_client = OpenAI(
    api_key="AIzaSyB6V3ajvZQy....", 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ==========================================
# 2. FUNGSI HELPER
# ==========================================

def get_db_connection():
    """Membuat koneksi ke database MySQL."""
    return mysql.connector.connect(**DB_CONFIG)

def check_auth():
    """Mengecek apakah user sudah login."""
    return 'user_id' in session

# ==========================================
# 3. ROUTE: AUTENTIKASI (LOGIN/REGISTER)
# ==========================================

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    nama = data.get("nama")
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Data tidak lengkap"}), 400
        
    hashed_pw = generate_password_hash(password)
    
    try:
        db = get_db_connection()
        cur = db.cursor()
        cur.execute("INSERT INTO users (nama, email, username, password) VALUES (%s, %s, %s, %s)", (nama, email, username, hashed_pw))
        db.commit()
        return jsonify({"message": "Registrasi berhasil, silakan login"})
    except mysql.connector.Error:
        return jsonify({"error": "Username sudah digunakan"}), 400
    finally:
        if 'cur' in locals(): cur.close()
        if 'db' in locals(): db.close()

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    db = get_db_connection()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    db.close()
    
    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({"message": "Login berhasil"})
    
    return jsonify({"error": "Username atau password salah"}), 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ==========================================
# 4. ROUTE: TAMPILAN HALAMAN (HTML VIEW)
# ==========================================

@app.route("/")
def home():
    if not check_auth():
        return redirect(url_for('login_page'))
    return render_template("index.html", username=session.get('username', 'User'))

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

# ==========================================
# 5. ROUTE: LOGIKA UTAMA RAG (DOKUMEN & CHAT)
# ==========================================

@app.route("/documents", methods=["GET"])
def list_documents():
    #Mengambil daftar dokumen milik user yang sedang login
    if not check_auth(): return jsonify({"error": "Unauthorized"}), 401
    
    db = get_db_connection()
    cur = db.cursor(dictionary=True)
    cur.execute("SELECT id, filename FROM documents WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
    docs = cur.fetchall()
    cur.close()
    db.close()
    
    return jsonify(docs)

@app.route("/upload", methods=["POST"])
def upload():
    #Proses upload PDF, ekstraksi teks, embedding, dan penyimpanan
    if not check_auth(): return jsonify({"error": "Unauthorized"}), 401

    try:
        file = request.files.get("file")
        if not file: return jsonify({"error": "No file"}), 400

        # 1. Simpan File PDF
        filename = file.filename
        pdf_path = os.path.join(STORAGE["pdf"], filename)
        file.save(pdf_path)

        # 2. Ekstrak Teks dari PDF
        doc = fitz.open(pdf_path)
        raw_text = ""
        for page in doc:
            raw_text += page.get_text("text") + " "
        doc.close()
        text = re.sub(r'\s+', ' ', raw_text).strip()
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(STORAGE["text"], txt_filename)

        # simpan text
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # 3. Chunking & Embedding
        chunk_size = 500
        overlap = 20 
        step = chunk_size - overlap
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), step)]

        # Proses embedding tetap sama
        embeddings = embedding_model.encode(chunks)
                    
        # 4. Simpan Chunks Teks ke File
        chunk_file = os.path.join(STORAGE["chunks"], f"{filename}.txt")
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n---CHUNK_SEP---\n")


        # 5. Simpan Vector Index (FAISS)
        faiss_path = os.path.join(STORAGE["faiss"], f"{filename}.index")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        faiss.write_index(index, faiss_path)

        # 6. Simpan Metadata ke Database
        db = get_db_connection()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO documents (filename, pdf_path, faiss_path, user_id) VALUES (%s, %s, %s, %s)",
            (filename, pdf_path, faiss_path, session['user_id'])
        )
        db.commit()
        cur.close()
        db.close()


        return jsonify({"message": f"Berhasil memproses {filename}", "jumlChunks" : len(chunks),
                        "jmlChar":len(text)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    if not check_auth(): return jsonify({"error": "Unauthorized"}), 401

    try:
        question = request.form.get("question")
        # Frontend akan mengirim array ID dalam bentuk JSON string
        doc_ids_json = request.form.get("document_ids") 

        if not doc_ids_json:
             return jsonify({"error": "Pilih minimal satu dokumen"}), 400

        # Convert string JSON "[1, 3]" menjadi list Python [1, 3]
        doc_ids = json.loads(doc_ids_json) 
        
        if not doc_ids:
            return jsonify({"error": "Pilih minimal satu dokumen"}), 400

        # 1. Ambil data semua dokumen yang dipilih dari Database
        db = get_db_connection()
        cur = db.cursor(dictionary=True)
        
        # Buat placeholder dinamis (%s, %s, ...) sesuai jumlah ID
        format_strings = ','.join(['%s'] * len(doc_ids))
        query = f"SELECT * FROM documents WHERE id IN ({format_strings}) AND user_id = %s"
        
        # Gabungkan doc_ids dengan user_id untuk parameter query
        params = tuple(doc_ids) + (session['user_id'],)
        
        cur.execute(query, params)
        docs_data = cur.fetchall()
        cur.close()
        db.close()

        if not docs_data:
            return jsonify({"error": "Dokumen tidak ditemukan"}), 404

        # 2. Embedding Pertanyaan
        q_embed = embedding_model.encode([question])
        q_embed_np = np.array(q_embed)

        all_candidates = [] # Penampung hasil pencarian dari semua file

        # 3. Looping: Cari di setiap dokumen
        for doc in docs_data:
            try:
                # Load Index FAISS
                index = faiss.read_index(doc['faiss_path'])
                
                # Load Text Chunks
                chunk_file = os.path.join(STORAGE["chunks"], f"{doc['filename']}.txt")
                with open(chunk_file, "r", encoding="utf-8") as f:
                    file_chunks = f.read().split("\n---CHUNK_SEP---\n")

                # Cari Top 3 di file INI
                D, I = index.search(q_embed_np, k=3)
                
                # Masukkan ke penampung global
                # D[0][i] adalah jarak (score), semakin kecil = semakin mirip
                for j, idx in enumerate(I[0]):
                    if idx < len(file_chunks):
                        all_candidates.append({
                            "score": D[0][j], 
                            "text": file_chunks[idx],
                            "source": doc['filename']
                        })
            except Exception as e:
                print(f"Skip file {doc['filename']} error: {e}")
                continue

        # 4. Global Ranking (Urutkan kandidat dari semua file)
        # Sort berdasarkan score terendah (jarak terdekat)
        all_candidates.sort(key=lambda x: x["score"])
        
        # Ambil 5 chunk terbaik dari gabungan semua file
        top_results = all_candidates[:5]

        if not top_results:
             return jsonify({"answer": "Tidak ditemukan informasi relevan.", "context": ""})

        # Susun Konteks untuk AI
        context_str = ""
        for item in top_results:
            context_str += f"[Sumber: {item['source']}]\n{item['text']}\n\n"
        context_str = ""
                
        for i, item in enumerate(top_results):
            # Format 4 angka belakang koma
            score = f"{item['score']:.4f}" 
            
            # Tambahkan ke string konteks yang akan dibaca AI & User
            # Kita taruh Score di paling depan agar terlihat jelas
            context_str += f"[Jarak: {score}] [Sumber: {item['source']}]\n{item['text']}\n\n"
            
            # Print ke terminal untuk pencatatan data penelitian
            print(f"Rank {i+1} | Score: {score} | File: {item['source']}")
        # 5. Kirim ke AI
        prompt = f"Jawab pertanyaan berdasarkan konteks gabungan berikut.\n\nKonteks:\n{context_str}\n\nPertanyaan: {question}"
        
        response = ai_client.chat.completions.create(
            model="gemini-2.5-flash", 
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content  

        # 6. Simpan Riwayat
        # Jika multi-file, document_id kita set NULL (None) agar netral
        # Atau jika cuma 1 file, simpan ID-nya
        saved_doc_id = doc_ids[0] if len(doc_ids) == 1 else None
        
        db = get_db_connection()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO qa_history (document_id, user_id, question, answer) VALUES (%s, %s, %s, %s)",
            (saved_doc_id, session['user_id'], question, answer)
        )
        db.commit()
        cur.close()
        db.close()

        return jsonify({"answer": answer, "context": context_str})
    
    except Exception as e:
        print(f"ERROR: {e}") # Print error di terminal backend
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def get_history():
    """Mengambil SEMUA riwayat tanya jawab user tanpa filter dokumen."""
    if not check_auth(): return jsonify([]), 401
        
    try:
        db = get_db_connection()
        cur = db.cursor(dictionary=True)
        
        # Query simpel: Ambil semua punya user ini, urutkan dari yang terbaru
        query = """
            SELECT question, answer, created_at 
            FROM qa_history 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """
        
        cur.execute(query, (session['user_id'],))
        history = cur.fetchall()
        cur.close()
        db.close()
        
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 6. MENJALANKAN APLIKASI
# ==========================================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8000, debug=True)
