import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer, util
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import difflib

# --- Settings ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = "1i9wMQFOZwZ5ZzbKzjA1aT2yebnBpXVvf2eqCt1foCgo"
THRESHOLD = 0.80  # Passing score

# --- Google Sheets setup ---
@st.cache_resource
def get_gsheet():
    """Authorizes with Google Sheets and returns the sheet object."""
    try:
        # Check for local credentials.json first
        if "credentials.json" in os.listdir():
            creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
        # Fallback to Streamlit secrets
        elif st.secrets.get("gcp_service_account"):
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes=SCOPES
            )
        else:
            st.warning("Google credentials not configured.")
            return None
        client = gspread.authorize(creds)
        return client.open_by_key(SPREADSHEET_ID).sheet1
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

sheet = get_gsheet()

# --- Load sentence transformer model ---
@st.cache_resource
def load_model():
    """Loads the sentence transformer model for similarity scoring."""
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

model = load_model()

# --- Load translations ---
def load_translations(file_path="data/translations.json"):
    """Loads translation data from a JSON file."""
    if not os.path.exists(file_path):
        st.error("Translation file not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

translations = load_translations()
japanese_to_entry = {entry["japanese"]: entry for entry in translations}

# --- Helper functions ---
def compute_score_and_best(user_text, variants):
    """Computes similarity score between user input and correct translations."""
    embeddings = model.encode([user_text] + variants, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embeddings[0], embeddings[1:])
    best_idx = scores.argmax().item()
    best_score = scores.max().item()
    return best_score, variants[best_idx]

# --- Streamlit UI ---
st.title("🧠 Japanese to English Translation Helper")

if not japanese_to_entry:
    st.warning("No translation data found.")
    st.stop()

# --- Student info ---
st.markdown("### 👤 Student Information")
student_name = st.text_input("Your Name")
student_number = st.text_input("Student Number")

# --- Initialize session state ---
if "last_score" not in st.session_state:
    st.session_state.last_score = None
if "last_variant" not in st.session_state:
    st.session_state.last_variant = None
if "scores" not in st.session_state:
    st.session_state.scores = []
if "attempted_questions" not in st.session_state:
    st.session_state.attempted_questions = 0

# --- Sentence selection ---
selected_japanese = st.selectbox("Select a Japanese sentence:", list(japanese_to_entry.keys()))
entry = japanese_to_entry[selected_japanese]

# --- Translation input ---
st.markdown("### ✍️ Enter your English translation")
user_input = st.text_input("Your answer:")

# --- Try Translation ---
if st.button("🔍 Try Translation"):
    if user_input.strip() == "":
        st.warning("Please enter a translation before trying.")
    else:
        all_variants = [entry["english"]] + entry.get("alternatives", [])
        best_score, best_variant = compute_score_and_best(user_input, all_variants)
        st.session_state.last_score = best_score
        st.session_state.last_variant = best_variant
        st.session_state.attempted_questions += 1
        
        # Display the number of questions attempted
        st.markdown(f"**Questions attempted:** {st.session_state.attempted_questions}")

        # Display only the score
        if best_score >= THRESHOLD:
            st.success(f"✅ Good enough! Score: {best_score:.2f}")
        else:
            st.warning(f"⚠️ Not quite there yet. Score: {best_score:.2f}")

# --- Submit Translation ---
if st.button("✅ Submit this translation"):
    if student_name.strip() == "" or student_number.strip() == "":
        st.warning("Please enter your name and student number before submitting.")
    elif user_input.strip() == "":
        st.warning("Please enter a translation before submitting.")
    else:
        # Calculate score if "Try Translation" hasn't been clicked first
        if st.session_state.last_score is None:
            all_variants = [entry["english"]] + entry.get("alternatives", [])
            best_score, best_variant = compute_score_and_best(user_input, all_variants)
        else:
            best_score = st.session_state.last_score

        st.session_state.scores.append(best_score)
        st.success(f"✅ Translation score recorded locally: {best_score:.2f}")

        # Reset session state for the next question
        st.session_state.last_score = None
        st.session_state.last_variant = None

# --- Finish Session ---
if st.button("🏁 Finish Session and Submit Average Score"):
    if student_name.strip() == "" or student_number.strip() == "":
        st.warning("Please enter your name and student number before finishing session.")
    elif not st.session_state.scores:
        st.warning("No scores recorded this session.")
    else:
        average_score = sum(st.session_state.scores) / len(st.session_state.scores)
        number_of_questions = len(st.session_state.scores)
        if sheet is None:
            st.error("Google Sheet not available. Submission failed.")
        else:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row([
                    timestamp,
                    student_name,
                    student_number,
                    f"{average_score:.4f}",
                    "Pass" if average_score >= THRESHOLD else "Fail",
                    number_of_questions
                ])
                st.success(f"🏁 Session finished! Average score {average_score:.2f} submitted.")
                # Clear session state for a new session
                st.session_state.scores = []
                st.session_state.attempted_questions = 0
            except Exception as e:
                st.error(f"Failed to save your session summary: {e}")
