"""
ml_engine.py
The "AI Decision Engine" stage.

Screens on TWO cognitive domains with established links to ADHD in the
literature (Barkley's executive-function model, among others):
  - ATTENTION  -> checklist score + reaction-time speed/variability
  - MEMORY     -> a short working-memory (sequence recall) task

IMPORTANT: there is no public labeled
dataset of child ADHD-screening-to-reading-profile mappings, so this trains
an MLPClassifier on a SYNTHETIC dataset built from documented clinical
heuristics. This is clearly disclosed as a limitation and a stated
direction for future work (real clinical data / teacher-rated validation)
in the thesis. The point being demonstrated here is the end-to-end
AI-driven pipeline architecture, not a validated diagnostic model.

NOTE ON AGE VS. GRADE: age is used here because ADHD presentation and
research screening instruments are organized by age/developmental stage,
not by school grade/class. "Grade" elsewhere in this app is used ONLY to
select an age-appropriate reading passage -- it is never a screening or
classification input.
"""
import numpy as np
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "attention_model.joblib"
SCALER_PATH = "attention_scaler.joblib"

PROFILES = ["Steady Reader", "Visual-Dominant Learner", "Short-Burst Learner"]

PROFILE_SETTINGS = {
    "Steady Reader": {
        "chunk_sentences": 3,
        "font_size_px": 22,
        "use_audio": False,
        "session_minutes": 20,
        "break_every_chunks": 6,
        "bg_tint": "#FFFFFF",
        "description": "Sustains attention and recall well over longer passages. Standard pacing.",
        "indicative_level": "Low indicative signals",
    },
    "Visual-Dominant Learner": {
        "chunk_sentences": 2,
        "font_size_px": 26,
        "use_audio": True,
        "session_minutes": 12,
        "break_every_chunks": 4,
        "bg_tint": "#FFF7E6",
        "description": "Engages best with narration + highlighted text together. Moderate chunking supports recall.",
        "indicative_level": "Moderate indicative signals",
    },
    "Short-Burst Learner": {
        "chunk_sentences": 1,
        "font_size_px": 30,
        "use_audio": True,
        "session_minutes": 8,
        "break_every_chunks": 2,
        "bg_tint": "#EAF4FF",
        "description": "Needs very short bursts, frequent breaks, and strong audio/visual cues to support both attention and recall.",
        "indicative_level": "High indicative signals",
    },
}


def _generate_synthetic_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        profile = rng.choice(PROFILES, p=[0.4, 0.32, 0.28])

        if profile == "Steady Reader":
            screening_score = rng.normal(8, 3)
            avg_rt = rng.normal(480, 60)
            rt_var = rng.normal(60, 20)
            memory_score = rng.normal(0.82, 0.12)
        elif profile == "Visual-Dominant Learner":
            screening_score = rng.normal(15, 3)
            avg_rt = rng.normal(560, 70)
            rt_var = rng.normal(110, 25)
            memory_score = rng.normal(0.60, 0.15)
        else:  # Short-Burst Learner
            screening_score = rng.normal(22, 3)
            avg_rt = rng.normal(680, 90)
            rt_var = rng.normal(180, 35)
            memory_score = rng.normal(0.38, 0.15)

        age = rng.integers(6, 13)
        screening_score = float(np.clip(screening_score, 0, 30))
        avg_rt = float(np.clip(avg_rt, 300, 1000))
        rt_var = float(np.clip(rt_var, 20, 300))
        memory_score = float(np.clip(memory_score, 0, 1))

        X.append([screening_score, avg_rt, rt_var, memory_score, age])
        y.append(profile)

    return np.array(X), np.array(y)


def train_and_save_model():
    X, y = _generate_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(X_train_s, y_train)
    test_acc = clf.score(X_test_s, y_test)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return test_acc


def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        train_and_save_model()
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


def predict_profile(screening_score, avg_rt, rt_var, memory_score, age):
    clf, scaler = load_model()
    X = np.array([[screening_score, avg_rt, rt_var, memory_score, age]])
    X_s = scaler.transform(X)
    proba = clf.predict_proba(X_s)[0]
    idx = int(np.argmax(proba))
    profile = clf.classes_[idx]
    confidence = float(proba[idx])
    settings = PROFILE_SETTINGS[profile]
    return profile, confidence, settings

