"""
app.py
Demo: AI-assisted early screening & personalized adaptive reading platform
for children with attention difficulties.

Run with:  streamlit run app.py

This is a SEMINAR/DEFENSE DEMO, not a clinical or production system.
Every "AI" stage is real and runs live (an MLPClassifier trained at
startup), but the screening instrument is illustrative, not a validated
clinical tool, and the training data is synthetic. State this plainly
during the defense -- it is a strength (transparent, reproducible,
honest about scope) not a weakness.
"""
import io
import os
import random
import tempfile
import time

import pandas as pd
import streamlit as st

import db
import ml_engine
from content import get_passages_for_grade

st.set_page_config(page_title="Adaptive Reading Companion", layout="wide", page_icon="📖")
db.init_db()
db.seed_demo_data()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "active_child_id": None,
    "rt_round": 0,
    "rt_times": [],
    "game_stage": "idle",
    "ready_time": None,
    "mem_round": 0,
    "mem_correct": [],
    "mem_stage": "idle",
    "mem_sequence": [],
    "mem_user_sequence": [],
    "reading_chunk_idx": 0,
    "chunk_start_time": None,
    "chunk_times": [],
    "breaks_triggered": 0,
    "current_passage": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_screening_state():
    st.session_state.rt_round = 0
    st.session_state.rt_times = []
    st.session_state.game_stage = "idle"
    st.session_state.ready_time = None
    st.session_state.mem_round = 0
    st.session_state.mem_correct = []
    st.session_state.mem_stage = "idle"
    st.session_state.mem_sequence = []
    st.session_state.mem_user_sequence = []


def reset_reading_state():
    st.session_state.reading_chunk_idx = 0
    st.session_state.chunk_start_time = None
    st.session_state.chunk_times = []
    st.session_state.breaks_triggered = 0
    st.session_state.current_passage = None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📖 Adaptive Reading Companion")
st.sidebar.caption("MSc demo — AI-driven early screening & personalized reading")

children = db.list_children()
child_names = {c["id"]: f"{c['name']} ({c['grade']})" for c in children}

page = st.sidebar.radio(
    "Pipeline stage",
    [
        "1. User Registration",
        "2. Early Screening",
        "3. AI Decision Engine & Profile",
        "4. Personalized Reading + Adaptive Module",
        "5. Reading Assessment",
        "6. Progress Monitoring",
        "7. AI Recommendations",
        "8. Teacher & Parent Dashboard",
        "9. Reports",
    ],
)

if children:
    st.sidebar.divider()
    selected = st.sidebar.selectbox(
        "Active learner",
        options=list(child_names.keys()),
        format_func=lambda x: child_names[x],
        index=list(child_names.keys()).index(st.session_state.active_child_id)
        if st.session_state.active_child_id in child_names
        else 0,
    )
    st.session_state.active_child_id = selected

# ---------------------------------------------------------------------------
# 1. User Registration
# ---------------------------------------------------------------------------
if page == "1. User Registration":
    st.header("User Registration")
    st.write("Register a learner. In production this would collect guardian consent too.")

    with st.form("register_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Child's name")
            age = st.number_input(
                "Age", min_value=5, max_value=14, value=8,
                help="Age is the demographic variable used in screening/classification below.",
            )
        with col2:
            grade = st.selectbox(
                "Reading level (for content matching only)",
                ["Beginner Reader", "Developing Reader", "Fluent Reader"],
                help="This ONLY selects which passages are age/ability-appropriate. "
                     "It plays no role in the attention/memory screening — ADHD "
                     "presentation is not organized by school grade or class.",
            )
            guardian_name = st.text_input("Parent/Teacher name")
        guardian_role = st.radio("Registering as", ["Parent", "Teacher"], horizontal=True)
        submitted = st.form_submit_button("Register learner")

    if submitted and name:
        child_id = db.add_child(name, age, grade, guardian_name, guardian_role)
        st.session_state.active_child_id = child_id
        st.success(f"{name} registered! Go to 'Early Screening' next.")
        st.rerun()

    if children:
        st.subheader("Registered learners")
        st.dataframe(pd.DataFrame(children)[["name", "age", "grade", "guardian_name", "created_at"]])

# ---------------------------------------------------------------------------
# 2. Early Screening
# ---------------------------------------------------------------------------
elif page == "2. Early Screening":
    st.header("Early Screening Module")

    if not st.session_state.active_child_id:
        st.warning("Register or select a learner first.")
        st.stop()

    child = db.get_child(st.session_state.active_child_id)
    st.write(f"Screening: **{child['name']}** ({child['grade']}, age {child['age']})")

    st.markdown("#### Part A: Behavior checklist")
    st.caption(
        "Answered by parent/teacher. Adapted from common inattention indicators — "
        "illustrative for this demo, not a validated clinical instrument."
    )
    checklist_items = [
        "Struggles to stay focused on a single reading task",
        "Is easily distracted by sounds or movement nearby",
        "Loses track of place while reading aloud",
        "Finishes tasks noticeably slower than peers",
        "Needs instructions repeated often",
        "Fidgets or moves around during quiet activities",
        "Skips or rushes through comprehension questions",
        "Prefers pictures/audio over long blocks of text",
        "Attention improves noticeably after a short break",
        "Reacts strongly to bright colors or lively narration",
    ]
    scores = []
    for i, item in enumerate(checklist_items):
        scores.append(
            st.slider(item, 0, 3, 1, key=f"chk_{i}", help="0 = Never, 3 = Very often")
        )
    screening_score = sum(scores)
    st.info(f"Checklist score: **{screening_score} / 30**")

    st.markdown("#### Part B: Attention task (reaction time)")
    st.caption(
        "The child clicks the button as soon as it turns red. "
        "We measure average speed and *consistency* — variability is a well-known attention proxy."
    )

    ROUNDS = 5
    if st.session_state.rt_round < ROUNDS:
        st.write(f"Round {st.session_state.rt_round + 1} of {ROUNDS}")

        if st.session_state.game_stage == "idle":
            if st.button("Start round", key=f"start_{st.session_state.rt_round}"):
                st.session_state.game_stage = "waiting"
                st.rerun()

        elif st.session_state.game_stage == "waiting":
            delay = random.uniform(1.2, 3.0)
            with st.spinner("Wait for it..."):
                time.sleep(delay)
            st.session_state.ready_time = time.time()
            st.session_state.game_stage = "ready"
            st.rerun()

        elif st.session_state.game_stage == "ready":
            if st.button("🔴 CLICK NOW!", key=f"click_{st.session_state.rt_round}"):
                rt_ms = (time.time() - st.session_state.ready_time) * 1000
                st.session_state.rt_times.append(rt_ms)
                st.session_state.rt_round += 1
                st.session_state.game_stage = "idle"
                st.rerun()

        st.caption("Having trouble with the timing? You can skip this and use typical values instead:")
        if st.button("⏭️ Skip live test (use typical calibration)", key=f"skip_{st.session_state.rt_round}"):
            st.session_state.rt_times = [550, 580, 520, 610, 540]
            st.session_state.rt_round = ROUNDS
            st.session_state.game_stage = "idle"
            st.rerun()
    else:
        avg_rt = sum(st.session_state.rt_times) / len(st.session_state.rt_times)
        rt_var = pd.Series(st.session_state.rt_times).std() or 0.0
        st.success(f"Done! Avg reaction time: {avg_rt:.0f} ms · Variability: {rt_var:.0f} ms")
        if st.button("Redo reaction task"):
            st.session_state.rt_round = 0
            st.session_state.rt_times = []
            st.session_state.game_stage = "idle"
            st.rerun()

    st.markdown("#### Part C: Memory task (sequence recall)")
    st.caption(
        "The child is shown a short sequence of icons, then recreates it from memory. "
        "Working memory is a second, well-documented cognitive domain linked to ADHD "
        "presentation alongside attention — screening both improves accuracy over "
        "using attention alone."
    )

    MEM_ROUNDS = 3
    MEM_ICONS = ["🔴", "🔵", "🟢", "🟡", "⭐", "🔶"]
    MEM_LENGTHS = [3, 4, 5]

    if st.session_state.mem_round < MEM_ROUNDS:
        length = MEM_LENGTHS[st.session_state.mem_round]
        st.write(f"Round {st.session_state.mem_round + 1} of {MEM_ROUNDS} (sequence length {length})")

        if st.session_state.mem_stage == "idle":
            if st.button("Show sequence", key=f"memstart_{st.session_state.mem_round}"):
                st.session_state.mem_sequence = random.sample(MEM_ICONS, length)
                st.session_state.mem_user_sequence = []
                st.session_state.mem_stage = "show"
                st.rerun()

        elif st.session_state.mem_stage == "show":
            st.markdown(
                f"<h2 style='letter-spacing:12px;'>{' '.join(st.session_state.mem_sequence)}</h2>",
                unsafe_allow_html=True,
            )
            with st.spinner("Memorize this sequence..."):
                time.sleep(1.5 + 0.6 * length)
            st.session_state.mem_stage = "recall"
            st.rerun()

        elif st.session_state.mem_stage == "recall":
            st.write("Now click the icons in the same order:")
            cols = st.columns(len(MEM_ICONS))
            for i, icon in enumerate(MEM_ICONS):
                if cols[i].button(icon, key=f"memicon_{st.session_state.mem_round}_{i}"):
                    st.session_state.mem_user_sequence.append(icon)
                    st.rerun()

            st.write("Your answer so far:", " ".join(st.session_state.mem_user_sequence) or "—")

            if st.session_state.mem_user_sequence:
                if st.button("Clear", key=f"memclear_{st.session_state.mem_round}"):
                    st.session_state.mem_user_sequence = []
                    st.rerun()

            if len(st.session_state.mem_user_sequence) == length:
                correct = st.session_state.mem_user_sequence == st.session_state.mem_sequence
                st.session_state.mem_correct.append(correct)
                st.session_state.mem_round += 1
                st.session_state.mem_stage = "idle"
                st.rerun()

        st.caption("Having trouble? Skip and use typical values instead:")
        if st.button("⏭️ Skip memory task (use typical calibration)", key="mem_skip"):
            st.session_state.mem_correct = [True, True, False]
            st.session_state.mem_round = MEM_ROUNDS
            st.session_state.mem_stage = "idle"
            st.rerun()
    else:
        memory_score = sum(st.session_state.mem_correct) / MEM_ROUNDS
        st.success(f"Done! Recalled {sum(st.session_state.mem_correct)}/{MEM_ROUNDS} sequences correctly.")
        if st.button("Redo memory task"):
            st.session_state.mem_round = 0
            st.session_state.mem_correct = []
            st.session_state.mem_stage = "idle"
            st.rerun()

    both_done = st.session_state.rt_round >= ROUNDS and st.session_state.mem_round >= MEM_ROUNDS
    if both_done:
        avg_rt = sum(st.session_state.rt_times) / len(st.session_state.rt_times)
        rt_var = pd.Series(st.session_state.rt_times).std() or 0.0
        memory_score = sum(st.session_state.mem_correct) / MEM_ROUNDS

        st.divider()
        if st.button("➡️ Submit screening & run AI Decision Engine", type="primary"):
            profile, confidence, settings = ml_engine.predict_profile(
                screening_score, avg_rt, rt_var, memory_score, child["age"]
            )
            db.save_screening(
                child["id"], screening_score, avg_rt, rt_var, memory_score,
                profile, confidence, settings,
            )
            reset_screening_state()
            st.success("Screening saved. Go to 'AI Decision Engine & Profile'.")
    else:
        st.info("Complete both the attention task and the memory task above to continue.")

# ---------------------------------------------------------------------------
# 3. AI Decision Engine & Learner Profile
# ---------------------------------------------------------------------------
elif page == "3. AI Decision Engine & Profile":
    st.header("AI Decision Engine & Learner Profile Creation")

    if not st.session_state.active_child_id:
        st.warning("Register or select a learner first.")
        st.stop()

    child = db.get_child(st.session_state.active_child_id)
    latest = db.get_latest_screening(child["id"])

    if not latest:
        st.warning("No screening on file yet. Complete 'Early Screening' first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Checklist score", f"{latest['screening_score']}/30")
    c2.metric("Avg reaction time", f"{latest['avg_reaction_time']:.0f} ms")
    c3.metric("Reaction variability", f"{latest['reaction_variability']:.0f} ms")
    c4.metric("Memory score", f"{(latest['memory_score'] or 0)*100:.0f}%")

    st.subheader(f"🧠 Predicted attention profile: **{latest['attention_profile']}**")
    st.caption(
        f"Screening indicator: **{latest['settings'].get('indicative_level', 'n/a')}** "
        "— a screening signal to guide content adaptation, not a diagnosis."
    )
    st.progress(latest["confidence"], text=f"Model confidence: {latest['confidence']*100:.1f}%")
    st.write(latest["settings"]["description"])

    st.markdown("##### Auto-generated content settings for this learner")
    s = latest["settings"]
    st.table(pd.DataFrame({
        "Setting": ["Chunk size", "Font size", "Audio narration", "Session length", "Break frequency"],
        "Value": [
            f"{s['chunk_sentences']} sentence(s) at a time",
            f"{s['font_size_px']} px",
            "On" if s["use_audio"] else "Off",
            f"{s['session_minutes']} minutes",
            f"Every {s['break_every_chunks']} chunks",
        ],
    }))

    st.caption(
        "Model: scikit-learn MLPClassifier, trained live on synthetic data reflecting "
        "documented screening-score / reaction-time relationships. See README for the "
        "validation limitation and future-work statement."
    )

# ---------------------------------------------------------------------------
# 5. Personalized Reading + Adaptive Learning Module
# ---------------------------------------------------------------------------
elif page == "4. Personalized Reading + Adaptive Module":
    st.header("Personalized Reading Content & Adaptive Learning Module")

    if not st.session_state.active_child_id:
        st.warning("Register or select a learner first.")
        st.stop()

    child = db.get_child(st.session_state.active_child_id)
    latest = db.get_latest_screening(child["id"])
    if not latest:
        st.warning("Complete screening first so content can be personalized.")
        st.stop()

    settings = latest["settings"]
    passages = get_passages_for_grade(child["grade"])

    if st.session_state.current_passage is None:
        titles = [p["title"] for p in passages]
        choice = st.selectbox("Choose a passage", titles)
        if st.button("Start reading session", type="primary"):
            st.session_state.current_passage = next(p for p in passages if p["title"] == choice)
            reset_reading_state()
            st.session_state.current_passage = next(p for p in passages if p["title"] == choice)
            st.session_state.chunk_start_time = time.time()
            st.rerun()
        st.stop()

    passage = st.session_state.current_passage
    chunk_size = settings["chunk_sentences"]
    sentences = passage["sentences"]
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    idx = st.session_state.reading_chunk_idx

    bg_tint = settings["bg_tint"]
    st.markdown(
        f"<div style='background-color:{bg_tint}; padding:20px; border-radius:10px;'>"
        f"<h3 style='color:#1a1a1a; margin:0;'>{passage['title']}</h3></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if idx < len(chunks):
        chunk_text = " ".join(chunks[idx])
        st.markdown(
            f"<p style='font-size:{settings['font_size_px']}px; line-height:1.6;'>{chunk_text}</p>",
            unsafe_allow_html=True,
        )

        if settings["use_audio"]:
            try:
                from gtts import gTTS
                tts = gTTS(chunk_text)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                st.audio(buf.getvalue(), format="audio/mp3")
            except Exception:
                st.caption("🔇 Audio narration unavailable right now (needs internet) — text-only mode.")

        st.caption(f"Chunk {idx + 1} of {len(chunks)}")

        if st.button("Next ➡️"):
            elapsed = time.time() - (st.session_state.chunk_start_time or time.time())
            st.session_state.chunk_times.append(elapsed)

            # Adaptive Learning Module: if this chunk took a long time,
            # treat it as a disengagement signal and insert a break.
            if elapsed > 25:
                st.session_state.breaks_triggered += 1
                st.session_state["_show_break"] = True

            st.session_state.reading_chunk_idx += 1
            st.session_state.chunk_start_time = time.time()
            st.rerun()

        if (idx + 1) % settings["break_every_chunks"] == 0 or st.session_state.get("_show_break"):
            st.info("⏸️ Adaptive break: take 15 seconds, then continue whenever ready.")
            st.session_state["_show_break"] = False
    else:
        st.success("Passage complete! Go to 'Reading Assessment' to check comprehension.")
        if st.button("🔁 Read a different passage"):
            reset_reading_state()
            st.rerun()

# ---------------------------------------------------------------------------
# 5. Reading Assessment
# ---------------------------------------------------------------------------
elif page == "5. Reading Assessment":
    st.header("7. Reading Assessment")

    if not st.session_state.active_child_id:
        st.warning("Register or select a learner first.")
        st.stop()
    if not st.session_state.current_passage or st.session_state.reading_chunk_idx == 0:
        st.warning("Complete a reading session in stage 4 first.")
        st.stop()

    passage = st.session_state.current_passage
    child = db.get_child(st.session_state.active_child_id)

    st.write(f"Comprehension check for **{passage['title']}**")
    answers = {}
    with st.form("quiz_form"):
        for i, q in enumerate(passage["quiz"]):
            answers[i] = st.radio(q["q"], q["options"], key=f"quiz_{i}", index=None)
        submitted = st.form_submit_button("Submit answers")

    if submitted:
        correct = sum(
            1 for i, q in enumerate(passage["quiz"]) if answers[i] == q["answer"]
        )
        score_pct = correct / len(passage["quiz"]) * 100
        avg_chunk_time = (
            sum(st.session_state.chunk_times) / len(st.session_state.chunk_times)
            if st.session_state.chunk_times else 0
        )
        db.save_session(
            child["id"], passage["title"], score_pct, avg_chunk_time,
            st.session_state.breaks_triggered,
        )
        st.success(f"Score: {correct}/{len(passage['quiz'])} ({score_pct:.0f}%) — saved to progress history.")
        reset_reading_state()

# ---------------------------------------------------------------------------
# 6. Progress Monitoring
# ---------------------------------------------------------------------------
elif page == "6. Progress Monitoring":
    st.header("Progress Monitoring")

    if not st.session_state.active_child_id:
        st.warning("Register or select a learner first.")
        st.stop()

    child = db.get_child(st.session_state.active_child_id)
    sessions = db.get_sessions(child["id"])

    if not sessions:
        st.info("No reading sessions logged yet for this learner.")
        st.stop()

    df = pd.DataFrame(sessions)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    df["session_number"] = range(1, len(df) + 1)

    if len(df) < 2:
        st.info(
            "Only one session logged so far — trend charts need at least 2 sessions "
            "to show a line. Here's the session on file:"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Comprehension score", f"{df['quiz_score'].iloc[0]:.0f}%")
        c2.metric("Avg time per chunk", f"{df['avg_time_per_chunk'].iloc[0]:.1f}s")
        c3.metric("Breaks triggered", int(df["breaks_triggered"].iloc[0]))
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Comprehension score over time")
            st.line_chart(df.set_index("session_number")["quiz_score"])
        with col2:
            st.subheader("Avg time per chunk (engagement proxy)")
            st.line_chart(df.set_index("session_number")["avg_time_per_chunk"])

    st.subheader("Session log")
    st.dataframe(df[["created_at", "passage_title", "quiz_score", "avg_time_per_chunk", "breaks_triggered"]])
# ---------------------------------------------------------------------------
# 7. AI Recommendations
# ---------------------------------------------------------------------------
elif page == "7. AI Recommendations":
    st.header("AI Recommendations")

    if not st.session_state.active_child_id:
        st.warning("Register or select a learner first.")
        st.stop()

    child = db.get_child(st.session_state.active_child_id)
    sessions = db.get_sessions(child["id"])
    latest_screen = db.get_latest_screening(child["id"])

    if not sessions or not latest_screen:
        st.info("Need at least one screening and one reading session to generate a recommendation.")
        st.stop()

    df = pd.DataFrame(sessions)
    recent = df.tail(3)
    avg_score = recent["quiz_score"].mean()
    trend = (
        recent["quiz_score"].iloc[-1] - recent["quiz_score"].iloc[0]
        if len(recent) > 1 else 0
    )
    avg_breaks = recent["breaks_triggered"].mean()

    st.metric("Avg recent comprehension score", f"{avg_score:.0f}%")
    st.metric("Trend (last vs first of recent sessions)", f"{trend:+.0f} pts")

    st.subheader("Recommendation")
    if avg_score < 50 or avg_breaks >= 2:
        st.warning(
            "📉 Reduce difficulty: shorter passages, smaller chunks, and more frequent "
            "breaks recommended for the next sessions."
        )
    elif avg_score >= 80 and trend >= 0:
        st.success(
            "📈 Learner is thriving: increase passage length slightly and/or move to the "
            "increase passage length slightly and/or move to the next reading level."
        )
    else:
        st.info("➡️ Maintain current profile settings; performance is stable.")

# ---------------------------------------------------------------------------
# 8. Teacher & Parent Dashboard
# ---------------------------------------------------------------------------
elif page == "8. Teacher & Parent Dashboard":
    st.header("Teacher & Parent Dashboard")

    if not children:
        st.info("No learners registered yet.")
        st.stop()

    for c in children:
        with st.expander(f"👤 {c['name']} — {c['grade']}, age {c['age']}"):
            latest = db.get_latest_screening(c["id"])
            sessions = db.get_sessions(c["id"])
            if latest:
                st.write(f"**Attention profile:** {latest['attention_profile']} "
                          f"(confidence {latest['confidence']*100:.0f}%)")
            else:
                st.write("No screening completed yet.")
            if sessions:
                df = pd.DataFrame(sessions)
                df["created_at"] = pd.to_datetime(df["created_at"])
                df = df.sort_values("created_at").reset_index(drop=True)
                st.write(f"**Sessions logged:** {len(df)} · "
                        f"**Avg score:** {df['quiz_score'].mean():.0f}%")
                if len(df) < 2:
                    st.caption(f"Latest score: {df['quiz_score'].iloc[-1]:.0f}% (need 2+ sessions for a trend line)")
                else:
                    df["session_number"] = range(1, len(df) + 1)
                    st.line_chart(df.set_index("session_number")["quiz_score"])
            else:
                st.write("No reading sessions yet.")
# ---------------------------------------------------------------------------
# 9. Reports
# ---------------------------------------------------------------------------
elif page == "9. Reports":
    st.header("Reports")

    if not st.session_state.active_child_id:
        st.warning("Register or select a learner first.")
        st.stop()

    child = db.get_child(st.session_state.active_child_id)
    latest = db.get_latest_screening(child["id"])
    sessions = db.get_sessions(child["id"])

    st.subheader(f"Report — {child['name']}")
    if latest:
        st.write(f"Attention profile: **{latest['attention_profile']}**")
    if sessions:
        df = pd.DataFrame(sessions)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV report", csv, file_name=f"{child['name']}_report.csv")
    else:
        st.info("No session data to report yet.")
