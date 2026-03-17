import os
import re
import json
import hashlib
import random
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Gemini client ──────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.5-flash"

# ── Halacha system prompt (kept small — document chunks go in user message) ──
SYSTEM_PROMPT = """אתה פוסק הלכה מומחה המתמחה אך ורק בספרי "ילקוט יוסף" של הרב עובדיה יוסף זצ"ל והרב יצחק יוסף שליט"א.

כללים מחייבים:
1. עליך לענות על שאלות אך ורק על סמך קטעי המסמך המצורפים. אין להשתמש בידע חיצוני, אין להוסיף לוגיקה משלך, ואין לשנות את משמעות הטקסט. "רק מה שכתוב".
2. תמיד ציין את המקור המדויק (פרק, סימן, סעיף, הלכה, עמוד וכו') מתוך הקטעים לצד תשובתך.
3. אם התשובה לא נמצאת בקטעים המצורפים, השב בדיוק: "התשובה לא נמצאת בקובץ המקור".
4. ענה בעברית בלבד.
5. השתמש בשפה הלכתית ברורה ומדויקת.
6. אם יש מחלוקת פוסקים המוזכרת בקטעים, ציין אותה כפי שמופיעה בטקסט.
7. אל תמציא מקורות או מראי מקומות שלא מופיעים בקטעים.
8. כאשר אתה מצטט, ציטוט מדויק ככל האפשר מהטקסט המקורי.

כללי עיצוב התשובה:
- השתמש ב-**כוכביות כפולות** להדגשת מונחים הלכתיים חשובים ומילות מפתח.
- חלק את התשובה לפסקאות ברורות ומסודרות.
- כאשר יש מספר דינים או פרטים, השתמש ברשימה ממוספרת (1. 2. 3.).
- בסוף התשובה, תמיד הוסף שורה ריקה ואז שורה שמתחילה בדיוק ב-"מקור:" ואחריה הציון המדויק של המקור מתוך הקטעים (לדוגמה: "מקור: סימן שב, סעיף א, עמוד 45")."""


# ══════════════════════════════════════════════════════════════════════════════
#  Feature #16: Hebrew stop-word removal
# ══════════════════════════════════════════════════════════════════════════════
HEBREW_STOP_WORDS = {
    "של", "את", "הוא", "היא", "אני", "על", "עם", "כי", "לא", "אם",
    "גם", "או", "כל", "אבל", "הם", "הן", "אנחנו", "יש", "אין",
    "רק", "עד", "מן", "בין", "כן", "אלא", "כמו", "לפי", "אותו",
    "אותה", "שלא", "אשר", "כאשר", "היה", "היו", "היתה", "יהיה",
    "להיות", "זה", "זו", "זאת", "אלה", "אלו", "כך", "כבר", "עוד",
    "מאד", "ביותר", "אחר", "אחרי", "לפני", "אצל", "בלי", "למה",
    "מה", "איך", "מי", "כמה", "מתי", "היכן", "איפה", "שם", "פה",
    "הנה", "בו", "בה", "בהם", "שהוא", "שהיא", "שהם", "ואם", "וכן",
}


def remove_stop_words(text):
    """Remove Hebrew stop words from text for better search quality."""
    words = re.findall(r'[\u0590-\u05FF\w]+', text)
    return ' '.join(w for w in words if w not in HEBREW_STOP_WORDS)


# ══════════════════════════════════════════════════════════════════════════════
#  Feature #18: Auto-detect Halacha topic
# ══════════════════════════════════════════════════════════════════════════════
HALACHA_TOPICS = {
    "שבת": {"keywords": ["שבת", "מוצאי שבת", "ערב שבת", "שביתה", "מלאכה", "מלאכות", "מוקצה", "הבדלה", "קידוש"], "icon": "🕯️"},
    "ברכות": {"keywords": ["ברכה", "ברכות", "ברכת", "שהכל", "העץ", "האדמה", "מזונות", "המוציא", "ברכת המזון"], "icon": "🙏"},
    "תפילה": {"keywords": ["תפילה", "תפילת", "שמונה עשרה", "עמידה", "שחרית", "מנחה", "ערבית", "מעריב", "קריאת שמע"], "icon": "📖"},
    "כשרות": {"keywords": ["כשר", "כשרות", "טרף", "בשר", "חלב", "בשרי", "חלבי", "הכשר", "שחיטה", "תערובות"], "icon": "🍽️"},
    "חגים ומועדים": {"keywords": ["פסח", "סוכות", "שבועות", "ראש השנה", "יום כיפור", "חנוכה", "פורים", "סוכה", "מצה", "שופר", "חג"], "icon": "📅"},
    "נידה וטהרה": {"keywords": ["נידה", "טהרה", "מקוה", "טבילה", "טהרת המשפחה"], "icon": "💧"},
    "שמיטה": {"keywords": ["שמיטה", "שביעית", "היתר מכירה", "אוצר בית דין"], "icon": "🌱"},
    "אבלות": {"keywords": ["אבלות", "אבל", "שבעה", "שלושים", "הספד", "קריעה", "ניחום אבלים"], "icon": "🕊️"},
    "נישואין": {"keywords": ["נישואין", "חתונה", "קידושין", "כתובה", "חופה"], "icon": "💍"},
    "מזוזה ותפילין": {"keywords": ["מזוזה", "תפילין", "ציצית", "מזוזות"], "icon": "📜"},
    "צדקה וחסד": {"keywords": ["צדקה", "מעשר", "חסד", "גמילות חסדים", "נתינה"], "icon": "🤲"},
    "בין אדם לחברו": {"keywords": ["לשון הרע", "רכילות", "הלבנת פנים", "גניבת דעת", "אונאה"], "icon": "🤝"},
}


def detect_topic(question):
    """Detect the halacha topic from a question. Returns (topic_name, icon) or (None, None)."""
    question_lower = question.strip()
    best_topic = None
    best_count = 0

    for topic, info in HALACHA_TOPICS.items():
        count = sum(1 for kw in info["keywords"] if kw in question_lower)
        if count > best_count:
            best_count = count
            best_topic = topic

    if best_topic and best_count > 0:
        return best_topic, HALACHA_TOPICS[best_topic]["icon"]
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
#  Feature #14: Query Cache (JSON file)
# ══════════════════════════════════════════════════════════════════════════════
CACHE_FILE = os.path.join(BASE_DIR, "query_cache.json")
query_cache = {}


def load_cache():
    """Load query cache from disk."""
    global query_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                query_cache = json.load(f)
            print(f"[✓] טעינת מטמון: {len(query_cache)} שאילתות")
        except Exception:
            query_cache = {}


def save_cache():
    """Persist query cache to disk."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(query_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_cache_key(question):
    """Generate a deterministic cache key for a question."""
    normalized = re.sub(r'\s+', ' ', question.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
#  Feature #19: Search Logging
# ══════════════════════════════════════════════════════════════════════════════
LOG_FILE = os.path.join(BASE_DIR, "search_logs.json")


def log_search(question, topic, found, chunks_used, cached):
    """Append a search event to search_logs.json."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "topic": topic,
        "answer_found": found,
        "chunks_used": chunks_used,
        "from_cache": cached,
    }
    try:
        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append(entry)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Feature #20: Enhanced source parser
# ══════════════════════════════════════════════════════════════════════════════
SOURCE_PATTERN = re.compile(
    r'(?:כרך|חלק)\s+[א-ת\d"\']+|'
    r'סימן\s+[א-ת\d"\']+|'
    r'סעיף\s+[א-ת\d"\']+|'
    r'הלכה\s+[א-ת\d"\']+|'
    r'עמוד\s+\d+|'
    r'עמ\'\s*\d+|'
    r'דף\s+[א-ת\d"\']+',
    re.MULTILINE,
)


def parse_source_structure(source_text):
    """Parse a source citation into structured parts (כרך, סימן, סעיף, etc.)."""
    parts = SOURCE_PATTERN.findall(source_text)
    if parts:
        return [p.strip() for p in parts]
    return [source_text.strip()] if source_text.strip() else []


# ══════════════════════════════════════════════════════════════════════════════
#  Load pre-built index from data/ (NO .docx processing at startup!)
# ══════════════════════════════════════════════════════════════════════════════

DOCUMENT_FILENAME = ""
DOCUMENT_CHAR_COUNT = 0
DOCUMENT_PAGE_COUNT = None
DOCUMENT_CHUNK_COUNT = 0

# These hold the pre-built data loaded from disk
chunks = []
word_vectorizer = None
word_matrix = None
char_vectorizer = None
char_matrix = None
daily_snippets = []


def load_prebuilt_index():
    """Load the pre-computed TF-IDF index and chunks from the data/ folder."""
    global chunks, word_vectorizer, word_matrix, char_vectorizer, char_matrix
    global daily_snippets
    global DOCUMENT_FILENAME, DOCUMENT_CHAR_COUNT, DOCUMENT_PAGE_COUNT, DOCUMENT_CHUNK_COUNT

    print(f"\n{'='*60}")
    print("[*] טוען אינדקס מובנה מתיקיית data/...")
    print(f"{'='*60}")

    try:
        # Load metadata
        metadata_path = os.path.join(DATA_DIR, "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        DOCUMENT_FILENAME = metadata["document_filename"]
        DOCUMENT_CHAR_COUNT = metadata["char_count"]
        DOCUMENT_PAGE_COUNT = metadata.get("page_count")
        DOCUMENT_CHUNK_COUNT = metadata["chunk_count"]

        # Load pre-built objects
        chunks = joblib.load(os.path.join(DATA_DIR, "chunks.joblib"))
        word_vectorizer = joblib.load(os.path.join(DATA_DIR, "word_vectorizer.joblib"))
        word_matrix = joblib.load(os.path.join(DATA_DIR, "word_matrix.joblib"))
        char_vectorizer = joblib.load(os.path.join(DATA_DIR, "char_vectorizer.joblib"))
        char_matrix = joblib.load(os.path.join(DATA_DIR, "char_matrix.joblib"))
        daily_snippets = joblib.load(os.path.join(DATA_DIR, "daily_snippets.joblib"))

        print(f"[✓] מסמך: {DOCUMENT_FILENAME}")
        print(f"    תווים: {DOCUMENT_CHAR_COUNT:,}")
        if DOCUMENT_PAGE_COUNT:
            print(f"    עמודים: {DOCUMENT_PAGE_COUNT}")
        print(f"    קטעים: {DOCUMENT_CHUNK_COUNT:,}")
        print(f"    קטעי הלכה יומית: {len(daily_snippets):,}")
        print(f"[✓] אינדקס נטען בהצלחה — צריכת זיכרון מינימלית")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"[!] שגיאה בטעינת אינדקס: {e}")
        print("[!] יש להריץ build_index.py לפני הפריסה!")
        DOCUMENT_CHUNK_COUNT = 0


def search_index(query, top_k=10):
    """Search the pre-built TF-IDF index. Same logic as the original RAGIndex.search()."""
    if word_vectorizer is None or word_matrix is None:
        return []

    # Feature #16: Remove stop words from query
    clean_query = remove_stop_words(query)

    # Word-level scores
    word_vec = word_vectorizer.transform([clean_query])
    word_scores = cosine_similarity(word_vec, word_matrix).flatten()

    # Feature #17: Character n-gram scores for fuzzy matching
    char_scores = np.zeros_like(word_scores)
    if char_vectorizer is not None and char_matrix is not None:
        char_vec = char_vectorizer.transform([query])  # use original query for char n-grams
        char_scores = cosine_similarity(char_vec, char_matrix).flatten()

    # Combine: 70% word + 30% char n-gram
    combined_scores = 0.7 * word_scores + 0.3 * char_scores

    # Get top-K indices, sorted by score descending
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(combined_scores[idx])
        if score < 0.01:
            break  # below noise floor — skip
        results.append({
            "chunk_id": int(idx),
            "score": round(score, 4),
            "text": chunks[idx],
        })

    return results


# Load once at startup — fast, memory-light
load_prebuilt_index()
load_cache()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    """Return info about the pre-loaded document so the frontend knows it is ready."""
    return jsonify({
        "ready": DOCUMENT_CHUNK_COUNT > 0,
        "filename": DOCUMENT_FILENAME,
        "char_count": DOCUMENT_CHAR_COUNT,
        "page_count": DOCUMENT_PAGE_COUNT,
        "chunk_count": DOCUMENT_CHUNK_COUNT,
    })


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    if not data:
        return jsonify({"error": "לא התקבלו נתונים", "error_type": "input"}), 400

    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "נא להזין שאלה", "error_type": "input"}), 400

    if DOCUMENT_CHUNK_COUNT == 0:
        return jsonify({"error": "לא נטען מסמך מקור. ודא שקובץ ילקוט יוסף נמצא בתיקיית הפרויקט.", "error_type": "system"}), 500

    # Feature #18: Detect topic
    topic_name, topic_icon = detect_topic(question)

    # Feature #14: Check cache first
    cache_key = get_cache_key(question)
    if cache_key in query_cache:
        cached = query_cache[cache_key]
        log_search(question, topic_name, cached.get("answer_found", True), cached.get("chunks_used", 0), True)
        return jsonify({
            "success": True,
            "answer": cached["answer"],
            "source_file": cached.get("source_file", DOCUMENT_FILENAME),
            "tokens_used": cached.get("tokens_used", {}),
            "chunks_used": cached.get("chunks_used", 0),
            "from_cache": True,
            "topic": topic_name,
            "topic_icon": topic_icon,
            "source_parts": cached.get("source_parts", []),
        })

    # ── Step 1: Retrieve relevant chunks locally (no API call) ────────────
    retrieved = search_index(question, top_k=10)

    if not retrieved:
        log_search(question, topic_name, False, 0, False)
        return jsonify({
            "success": True,
            "answer": "התשובה לא נמצאת בקובץ המקור",
            "source_file": DOCUMENT_FILENAME,
            "tokens_used": {},
            "topic": topic_name,
            "topic_icon": topic_icon,
            "source_parts": [],
        })

    # ── Step 2: Build a compact context from retrieved chunks ─────────────
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(f"[קטע {i} | רלוונטיות: {r['score']}]\n{r['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    user_message = (
        "להלן קטעים רלוונטיים מתוך ספרי ילקוט יוסף שנמצאו עבור השאלה:\n\n"
        "--- תחילת קטעים ---\n"
        f"{context_block}\n"
        "--- סוף קטעים ---\n\n"
        f"שאלה: {question}\n\n"
        "ענה אך ורק על סמך הקטעים שלמעלה. "
        "ציין מקור מדויק (סימן, סעיף, הלכה, עמוד). "
        'אם התשובה לא נמצאת בקטעים, השב: "התשובה לא נמצאת בקובץ המקור".'
    )

    # ── Step 3: Send only the small context to Gemini ─────────────────────
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=4096,
            ),
        )

        answer = response.text

        # Extract token usage
        tokens_used = {}
        usage = getattr(response, "usage_metadata", None)
        if usage:
            tokens_used = {
                "input": getattr(usage, "prompt_token_count", 0),
                "output": getattr(usage, "candidates_token_count", 0),
            }

        # Feature #20: Parse source structure from answer
        source_parts = parse_source_structure(answer)

        answer_found = "התשובה לא נמצאת בקובץ המקור" not in answer

        # Feature #14: Cache the result
        query_cache[cache_key] = {
            "answer": answer,
            "source_file": DOCUMENT_FILENAME,
            "tokens_used": tokens_used,
            "chunks_used": len(retrieved),
            "source_parts": source_parts,
            "answer_found": answer_found,
        }
        save_cache()

        # Feature #19: Log search
        log_search(question, topic_name, answer_found, len(retrieved), False)

        return jsonify({
            "success": True,
            "answer": answer,
            "source_file": DOCUMENT_FILENAME,
            "tokens_used": tokens_used,
            "chunks_used": len(retrieved),
            "from_cache": False,
            "topic": topic_name,
            "topic_icon": topic_icon,
            "source_parts": source_parts,
        })

    except Exception as e:
        error_msg = str(e)
        # Feature #8: Enhanced error classification
        if "API_KEY" in error_msg or "api_key" in error_msg or "401" in error_msg:
            return jsonify({"error": "מפתח API לא תקין. בדוק את הגדרות GEMINI_API_KEY", "error_type": "auth"}), 401
        elif "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return jsonify({"error": "חריגה ממגבלת בקשות. נסה שוב בעוד דקה", "error_type": "rate_limit"}), 429
        elif "block" in error_msg.lower() or "safety" in error_msg.lower():
            return jsonify({"error": "התוכן נחסם על ידי מסנני הבטיחות. נסה לנסח מחדש את השאלה.", "error_type": "safety"}), 400
        elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
            return jsonify({"error": "הבקשה ארכה יותר מדי זמן. נסה שוב.", "error_type": "timeout"}), 504
        elif "connect" in error_msg.lower() or "network" in error_msg.lower():
            return jsonify({"error": "שגיאת חיבור לשרת. בדוק את חיבור האינטרנט.", "error_type": "network"}), 503
        else:
            return jsonify({"error": f"שגיאה: {error_msg}", "error_type": "unknown"}), 500


@app.route("/daily", methods=["GET"])
def daily_halacha():
    """Return a complete, clean halachic paragraph with topic/siman context."""
    if not daily_snippets:
        return jsonify({"snippet": None, "topic": "", "siman": ""})

    # Deterministic seed from today's date → same snippet all day
    today_seed = int(datetime.now().strftime("%Y%m%d"))
    rng = random.Random(today_seed)
    entry = rng.choice(daily_snippets)

    return jsonify({
        "snippet": entry["text"],
        "topic": entry.get("topic", ""),
        "siman": entry.get("siman", ""),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  Feature #10: Feedback endpoint
# ══════════════════════════════════════════════════════════════════════════════
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.json")


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Record thumbs up/down feedback for a Q&A pair."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "לא התקבלו נתונים"}), 400

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": data.get("question", ""),
        "rating": data.get("rating", ""),  # "up" or "down"
    }

    try:
        feedback = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
                feedback = json.load(f)
        feedback.append(entry)
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)
        return jsonify({"success": True})
    except Exception:
        return jsonify({"error": "שגיאה בשמירת המשוב"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
