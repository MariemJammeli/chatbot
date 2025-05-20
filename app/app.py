import streamlit as st  # type: ignore
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder  # type: ignore
import google.generativeai as genai  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import joblib  # type: ignore
import os
import json
import zipfile

# ========== File Checks ==========
required_files = [
    'dataset.zip',
    'tech_recommendation_model.joblib',
    'tech_encoder.joblib',
    'download_model.py',
    'app.py',
    'error_type_encoder.joblib',
    'error_kind_encoder.joblib',
    'ml.csv'
]

current_files = os.listdir("app")
missing_files = [f for f in required_files if f not in current_files]

if missing_files:
    print("Fichiers manquants :", missing_files)
else:
    print("✅ Tous les fichiers requis sont présents.")

# ========== Chat History ==========
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ========== Session State ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None

# ========== Load data and models ==========
CSV_PATH = "app/ml.csv"
ZIP_PATH = "app/dataset.zip"

if not os.path.exists(CSV_PATH):
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("app")
        print("✅ Fichier ml.csv extrait depuis dataset.zip")
    else:
        raise FileNotFoundError("❌ Le fichier dataset.zip est introuvable.")

df = pd.read_csv(CSV_PATH, sep=';')
df["ErrorCode"] = df["ErrorKindTypeKey"].astype(str) + "-" + df["ErrorType"].astype(str)
df["text"] = df["Remark"].fillna("") + " " + df["ErrorMessage"].fillna("")
df = df.dropna(subset=["text", "ErrorCode", "RequiredOperations"])
df = df.reset_index(drop=True)

tech_encoder = joblib.load("app/tech_encoder.joblib")
error_kind_encoder = joblib.load("app/error_kind_encoder.joblib")
error_type_encoder = joblib.load("app/error_type_encoder.joblib")
lgb_model = joblib.load("app/tech_recommendation_model.joblib")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

df['PreferredTechnician'] = df['PreferredTechnician'].astype(str)
df['tech_encoded'] = tech_encoder.transform(df['PreferredTechnician'])

model_path = "Mariem23/DistillBertFinetuned"
tokenizer2 = AutoTokenizer.from_pretrained(model_path)
model2 = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2.to(device)
model2.eval()

label_encoder = LabelEncoder()
label_encoder.fit(df["ErrorCode"])
errorcode_to_operations = dict(zip(df["ErrorCode"], df["RequiredOperations"]))

genai.configure(api_key="AIzaSyAmWJigahLFZcanuotVI3eToMq_YP1M37c")
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# ========== Helper functions ==========
def generate_first_question(problem_text):
    prompt = (
        f"Ein Benutzer beschreibt folgendes technisches Problem: '{problem_text}'. "
        "Stelle eine präzise technische Frage auf Deutsch, um das Problem zu verstehen."
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def generate_second_question(problem_text, question1, answer1):
    prompt = (
        f"Ein Benutzer beschreibt folgendes technisches Problem: '{problem_text}'. "
        f"Er hat auf die erste Frage '{question1}' geantwortet mit: '{answer1}'. "
        "Stelle nun eine andere technische und vertiefende Folgefrage auf Deutsch, ohne die erste zu wiederholen."
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def predict_error_code_and_operations(answer_text):
    inputs = tokenizer2(answer_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model2(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    predicted_operations = errorcode_to_operations.get(predicted_label, "Unknown")
    return predicted_label, predicted_operations

def recommend_top_techs(job_row, tech_list, model, encoder, text_fields):
    job_text = ' '.join(str(job_row.get(field, '')) for field in text_fields)
    embedding = embedding_model.encode([job_text])[0]
    base_feats = np.array([error_kind_encoder.transform([str(job_row['ErrorKindTypeKey'])])[0],
                           error_type_encoder.transform([str(job_row['ErrorType'])])[0],
                           job_row['ExperienceInMonths']] + list(embedding), dtype=np.float32)

    candidates = []
    for tech in tech_list:
        tech_encoded = encoder.transform([tech])[0]
        input_vector = np.hstack([base_feats, tech_encoded])
        prob = model.predict_proba([input_vector])[0][1]
        candidates.append((tech, prob))

    seen = set()
    top3 = []
    for tech, score in sorted(candidates, key=lambda x: -x[1]):
        if tech not in seen:
            top3.append((tech, score))
            seen.add(tech)
        if len(top3) == 3:
            break

    return top3

# ========== Streamlit UI ==========
st.set_page_config(page_title="Techniker-Chatbot", layout="centered")
st.title("🤖 Technischer Support Chatbot")

# Sidebar history panel
st.sidebar.title("📂 Gesprächsverlauf")

# ➕ Button to start new conversation in sidebar
if st.sidebar.button("➕ Neue Konversation"):
    st.session_state.selected_history = None
    st.rerun()  # FIXED


# History buttons in sidebar
for idx, chat in enumerate(st.session_state.chat_history):
    if st.sidebar.button(chat["topic"], key=f"history_{idx}"):
        st.session_state.selected_history = idx
        st.rerun()
# Main conversation display
if st.session_state.selected_history is not None:
    selected = st.session_state.chat_history[st.session_state.selected_history]
    st.markdown("### 🗂️ Früheres Gespräch:")

    st.markdown(f"- **Fehlercode:** {selected['code']}")
    st.markdown(f"- **Operationen:** {selected['ops']}")
    st.markdown(f"- **Frage 1:** {selected['q1']}\n- **Antwort 1:** {selected['a1']}")
    st.markdown(f"- **Frage 2:** {selected['q2']}\n- **Antwort 2:** {selected['a2']}")

    st.markdown("## 👨‍🔧 Empfohlene Techniker:")
    if "technicians" in selected:
        for i, (tech, score) in enumerate(selected["technicians"], 1):
            st.info(f"{i}. {tech} (Vertrauensscore: {score:.4f})")

    if "estimated_time" in selected:
        st.markdown(f"⏱️ **Die Arbeit wird voraussichtlich etwa {selected['estimated_time']} Minuten dauern.**")

else:
    user_input = st.text_area("🔧 Beschreibe dein technisches Problem (auf Deutsch):")

    if user_input:
        st.markdown("## 🧠 Analyse läuft...")

        question1 = generate_first_question(user_input)
        st.markdown(f"**🤖 Frage 1:** {question1}")
        answer1 = st.text_input("✍️ Deine Antwort auf Frage 1:")

        if answer1:
            question2 = generate_second_question(user_input, question1, answer1)
            st.markdown(f"**🤖 Frage 2:** {question2}")
            answer2 = st.text_input("✍️ Deine Antwort auf Frage 2:")

            if answer2:
                all_answers = f"{answer1} {answer2}"
                code, ops = predict_error_code_and_operations(all_answers)

                st.markdown("## 🔍 Vorhersage:")
                st.success(f"**Fehlercode:** {code}")
                st.success(f"**Erforderliche Operationen:** {ops}")

                third_question = "Hat diese Information dein Problem gelöst?"
                st.markdown(f"**🤖 Frage 3:** {third_question}")
                solved_answer = st.text_input("✍️ Deine Antwort auf Frage 3:")

                if solved_answer and "nein" in solved_answer.lower():
                    availability = st.text_input("📅 Bitte gib dein verfügbares Datum ein (Format: TT.MM.JJJJ):")

                    if availability:
                        st.markdown(f"🗓️ **Verfügbarkeit notiert:** {availability}")

                        match = df[df["ErrorCode"] == code]
                        if not match.empty:
                            match_row = match.iloc[0]
                            job_row = {
                                "ErrorKindTypeKey": match_row["ErrorKindTypeKey"],
                                "ErrorType": match_row["ErrorType"],
                                "ExperienceInMonths": match_row.get("ExperienceInMonths", 0),
                                "text": all_answers,
                                "ErrorCode": code,
                                "RequiredOperations": ops
                            }

                            unique_techs = tech_encoder.inverse_transform(np.unique(df["tech_encoded"]))
                            text_fields = ["text", "ErrorCode", "RequiredOperations"]
                            top_techs = recommend_top_techs(job_row, unique_techs, lgb_model, tech_encoder, text_fields)

                            st.markdown("## 👨‍🔧 Top 3 empfohlene Techniker:")
                            technicians = [(tech, score) for tech, score in top_techs]
                            for i, (tech, score) in enumerate(technicians, 1):
                                st.info(f"{i}. {tech} (Vertrauensscore: {score:.4f})")

                            duration_match = df[(df["ErrorCode"] == code) & (df["RequiredOperations"] == ops)]
                            if not duration_match.empty:
                                estimated_time = int(duration_match["DurationInMinutes"].mean())
                                st.markdown(f"⏱️ **Die Arbeit wird voraussichtlich etwa {estimated_time} Minuten dauern.**")
                            else:
                                estimated_time = None
                                st.markdown("⏱️ **Keine Schätzung der Dauer verfügbar.**")

                            # Save conversation to history
                            new_history = {
                                "topic": f"Fehlercode {code} am {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                "code": code,
                                "ops": ops,
                                "q1": question1,
                                "a1": answer1,
                                "q2": question2,
                                "a2": answer2,
                                "technicians": technicians,
                                "estimated_time": estimated_time,
                            }
                            st.session_state.chat_history.append(new_history)
                            save_history(st.session_state.chat_history)
                        else:
                            st.markdown("⏱️ **Geschätzte Reparaturzeit:** Nicht verfügbar")
                    else:
                        st.warning("⚠️ Kein passender Fehlercode-Eintrag gefunden für die Technikerempfehlung.")
                elif solved_answer:
                    st.success("🎉 Danke, dass du den Chatbot benutzt hast! Wir freuen uns, dass dein Problem gelöst ist.")

# --- Always show this button at the bottom, outside all conditionals ---
st.markdown("---")
if st.button("🆕 Neue Konversation starten"):
    st.session_state.selected_history = None
    st.session_state.chat_history = []
    save_history(st.session_state.chat_history)
    st.rerun()  # FIXED


