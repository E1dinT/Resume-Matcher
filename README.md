# Resume Matcher (Backend-Focused)

A resume-job matching system that evaluates alignment based on **weighted skill importance and evidence strength**, and provides **actionable suggestions and automatic bullet rewrites**.

---

## 🚀 Features

- 📊 Weighted skill matching (not just keyword overlap)
- 🧠 Evidence-based scoring (distinguishes skills vs real experience)
- 📉 Gap analysis (high-priority missing concepts)
- 📝 Resume bullet rewriting (aligns resume with backend/infrastructure roles)
- 📦 FastAPI backend + simple frontend UI

---

## 🧠 How It Works

### 1. Concept Extraction
Extracts important concepts from job descriptions:
- Backend (API, services)
- Data (SQL, pipelines)
- Infra (cloud, Docker, CI/CD)

Each concept is weighted based on:
- Importance
- Section (requirements > nice-to-have)

---

### 2. Evidence Strength Scoring

Not all mentions are equal:

| Type | Score |
|------|------|
| Real project usage | 1.0 |
| Mention in bullet | 0.7 |
| Listed in skills | 0.35 |

---

### 3. Matching Score

Final score combines:

- Weighted keyword score (82%)
- Semantic similarity (18%)

---

### 4. Resume Optimization

Automatically rewrites resume bullets to:
- Add backend context
- Introduce API/database framing
- Improve alignment with job requirements

---

## 🛠 Tech Stack

- Python
- FastAPI
- Uvicorn
- HTML (frontend)

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
cd backend
python3 -m uvicorn main:app --reload