# Adaptive Reading Companion

**AI-assisted early screening and personalized adaptive reading for children
with attention difficulties (ADHD-focused).**

A working end-to-end prototype built to support an MSc Special Education
research project on the effect of AI-personalized reading content on
sustained attention in children with ADHD. Built as a single Streamlit
application with a live-trained scikit-learn classifier at its core.

> **Research prototype disclaimer:** the screening checklist here is
> illustrative and not a validated clinical instrument, and the classifier
> is trained on synthetic data reflecting documented screening-score /
> reaction-time / working-memory relationships from the literature — not
> real clinical outcomes. No public dataset currently links these variables
> directly; validating this pipeline against real classroom/clinical data
> is the proposed next research phase.
>
> **Age vs. grade:** ADHD presentation and screening research are organized
> by age/developmental stage, not by school grade or class. This app uses
> **age** as the demographic screening variable throughout. "Reading level"
> (Beginner/Developing/Fluent) exists purely to select age/ability-appropriate
> passages — it is never fed into the screening or classification logic.

---

## Pipeline

```
User Registration
    |
    v
Early Screening Module          (behavior checklist + attention task +
    |                               memory task)
    v
AI Decision Engine              (MLPClassifier → attention profile,
    |                              screened on BOTH attention and memory)
    v
Learner Profile Creation        (profile → content settings)
    |
    v
Personalised Reading Content    (passage bank, chunked per profile,
    |                             matched to reading level — not screening)
    v
Adaptive Learning Module        (live chunk-size + break adjustments)
    |
    v
Reading Assessment              (comprehension quiz)
    |
    v
Progress Monitoring             (score & engagement trends over time)
    |
    v
AI Recommendations              (rule engine on trend data)
    |
    v
Teacher & Parent Dashboards
    |
    v
Reports                         (CSV export)
```

## Tech stack

- **Frontend/app:** Streamlit
- **AI Decision Engine:** scikit-learn `MLPClassifier` (16→8 hidden layers), `StandardScaler`
  — trained on 5 features: checklist score, avg reaction time, reaction-time
  variability, working-memory (sequence-recall) score, and age
- **Persistence:** SQLite (zero external services required)
- **Audio narration:** gTTS (graceful fallback to text-only if offline)
- **Data:** pandas / numpy
- **Reports:** CSV export (native); PDF via `reportlab` (optional)

## Project structure

```
adhd_reading_app/
├── app.py              # Streamlit UI — all 9 pipeline stages
├── db.py                # SQLite schema, CRUD, demo-data seeding
├── ml_engine.py          # Synthetic dataset, MLPClassifier training & inference
├── content.py             # Sample reading passage bank + comprehension quizzes
├── train_model.py          # Standalone script to (re)train the classifier
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repo

```bash
git clone https://github.com/jamesdanas/adhd-reading-companion.git
cd adhd-reading-companion
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the AI Decision Engine

```bash
python3 train_model.py
```

This trains the `MLPClassifier` on the synthetic screening dataset and
saves `attention_model.joblib` + `attention_scaler.joblib` to the project
root, printing holdout accuracy (typically ~90–95%). This step is optional
— the app will auto-train on first launch if these files aren't present —
but running it explicitly lets you inspect the model before demoing.

### 5. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. A demo learner ("Udinah") is seeded
automatically on first launch so the Progress Monitoring, Recommendations,
and Dashboard pages have data to display immediately.

## Known limitations

- Screening checklist and memory task are illustrative, adapted from
  general inattention/working-memory indicators — not validated clinical
  instruments (e.g. not SNAP-IV/Conners or a standardized digit-span test as-is).
- Classifier is trained on synthetic data encoding documented heuristics
  across attention AND memory domains, not real longitudinal clinical/
  behavioral outcomes.
- Reaction-time and memory tasks run via Streamlit reruns — adequate for
  demonstration, not millisecond-precise like dedicated psychology testing
  software.
- "Reading level" is a content-matching convenience only; it is
  intentionally never used for screening or classification, since ADHD
  presentation does not correspond to school grade/class.
- Content library ships with 3 sample passages; a real deployment needs a
  larger, curriculum-aligned bank.
- SQLite is file-based and fine for a single-user demo; a multi-classroom
  deployment would need a proper multi-tenant database.

## Roadmap

- [ ] Replace synthetic training data with real (consented, anonymized)
      screening/session data once available
- [ ] Expand passage bank and support teacher-uploaded content
- [ ] Multi-user auth for real classroom/multi-child deployment
- [ ] Richer engagement signals (e.g. gaze/attention proxies beyond
      time-per-chunk) feeding back into the Adaptive Learning Module

## License

MIT

## Author

**James Danas** — AI/ML Engineer
GitHub: [@jamesdanas](https://github.com/jamesdanas)
Email: jamesdanas.y@gmail.com

Built in support of an MSc Special Education research project,
University of Jos.
