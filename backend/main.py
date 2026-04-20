from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Set, Tuple
import math
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str


class RewriteRequest(BaseModel):
    resume: str
    job_description: str


GENERIC_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "are", "from", "this", "have",
    "will", "our", "who", "what", "how", "into", "about", "looking", "role",
    "work", "team", "teams", "someone", "their", "them", "that", "than",
    "then", "also", "just", "very", "more", "most", "some", "such", "good",
    "well", "high", "strong", "basic", "ability", "able", "including",
    "required", "requirements", "nice", "join", "whatll", "youll", "we", "us",
    "they", "day", "one", "fast", "growing", "small", "real", "used", "using",
    "build", "built", "help", "helped", "helping", "improve", "improved",
    "maintain", "maintained", "familiar", "experience", "skills", "skill",
    "candidate", "position", "opportunity", "office", "remote", "present",
    "current", "currently", "based", "similar", "plus", "quickly", "clearly",
    "directly", "through", "while", "where", "when", "being", "takes", "take",
    "dont", "wait", "told", "pursuing", "completed", "recently", "day", "days",
    "month", "months", "year", "years", "preferred", "bonus", "ideal",
    "excellent", "great", "support", "supporting", "environment",
    "responsibility", "responsibilities", "intern", "internship", "full-time",
    "fulltime", "part-time", "parttime", "company", "platform", "product",
    "products", "business", "want", "need", "needs", "person", "job", "jobs",
    "etc", "a", "an", "to", "of", "in", "on", "at", "by", "as", "or", "be", "is"
}

SECTION_HEADERS = {
    "about the role": "role",
    "what you'll work on": "work",
    "what you’ll work on": "work",
    "requirements": "requirements",
    "nice to have": "nice_to_have",
    "why join us?": "benefits",
    "why join us": "benefits",
    "about novaflow": "about_company",
    "about the job": "about_job",
}

SECTION_WEIGHTS = {
    "requirements": 1.35,
    "work": 1.20,
    "role": 1.00,
    "nice_to_have": 0.65,
    "benefits": 0.15,
    "about_company": 0.20,
    "about_job": 0.20,
    "other": 0.60,
}

CANONICAL_ORDER = [
    "python", "go", "nodejs", "java", "c++",
    "backend", "api", "sql", "nosql", "database",
    "cloud", "aws", "gcp", "docker", "kubernetes", "terraform", "cicd", "monitoring",
    "pipelines", "data", "analysis", "integrations",
    "git", "linux", "communication", "degree", "startup", "biology",
]

CONCEPTS: Dict[str, Dict[str, object]] = {
    "python": {
        "aliases": ["python", "pandas", "numpy"],
        "category": "languages",
        "importance": 3.0,
    },
    "go": {
        "aliases": ["go", "golang"],
        "category": "languages",
        "importance": 2.4,
    },
    "nodejs": {
        "aliases": ["nodejs", "node.js", "node"],
        "category": "languages",
        "importance": 1.8,
    },
    "java": {
        "aliases": ["java"],
        "category": "languages",
        "importance": 1.6,
    },
    "c++": {
        "aliases": ["c++", "cpp"],
        "category": "languages",
        "importance": 1.5,
    },
    "backend": {
        "aliases": ["backend", "service", "services", "server", "microservice", "microservices"],
        "bridges": ["system architecture", "architecture"],
        "category": "backend_data",
        "importance": 2.8,
    },
    "api": {
        "aliases": ["api", "apis", "rest", "restful", "graphql", "endpoint", "endpoints", "api design"],
        "bridges": ["integrations", "third-party services"],
        "category": "backend_data",
        "importance": 2.9,
    },
    "sql": {
        "aliases": ["sql", "postgres", "postgresql", "mysql", "sqlite"],
        "category": "backend_data",
        "importance": 2.9,
    },
    "nosql": {
        "aliases": ["nosql", "mongodb", "redis", "dynamodb"],
        "category": "backend_data",
        "importance": 2.4,
    },
    "database": {
        "aliases": ["database", "databases", "data model", "data models", "schema", "schemas", "query", "queries", "table", "tables", "storage"],
        "category": "backend_data",
        "importance": 2.7,
    },
    "cloud": {
        "aliases": ["cloud", "cloud infrastructure", "infrastructure"],
        "bridges": ["deployment", "production"],
        "category": "infra_devops",
        "importance": 2.6,
    },
    "aws": {
        "aliases": ["aws", "amazon web services", "ec2", "s3", "lambda"],
        "category": "infra_devops",
        "importance": 2.5,
    },
    "gcp": {
        "aliases": ["gcp", "google cloud", "cloud run", "bigquery"],
        "category": "infra_devops",
        "importance": 2.5,
    },
    "docker": {
        "aliases": ["docker", "container", "containers", "containerization"],
        "category": "infra_devops",
        "importance": 2.1,
    },
    "kubernetes": {
        "aliases": ["kubernetes", "k8s"],
        "category": "infra_devops",
        "importance": 1.9,
    },
    "terraform": {
        "aliases": ["terraform", "infrastructure as code", "iac", "pulumi"],
        "category": "infra_devops",
        "importance": 1.8,
    },
    "cicd": {
        "aliases": ["ci/cd", "ci-cd", "cicd", "continuous integration", "continuous deployment", "github actions", "jenkins"],
        "category": "infra_devops",
        "importance": 2.0,
    },
    "monitoring": {
        "aliases": ["monitoring", "observability", "logging", "alerting", "tracing"],
        "category": "infra_devops",
        "importance": 1.9,
    },
    "pipelines": {
        "aliases": ["pipeline", "pipelines", "data pipeline", "data pipelines", "etl"],
        "category": "backend_data",
        "importance": 2.5,
    },
    "data": {
        "aliases": ["data", "dataset", "datasets"],
        "category": "backend_data",
        "importance": 1.8,
    },
    "analysis": {
        "aliases": ["analysis", "analytics", "analyze", "analyst", "insights", "evaluation", "visualization", "visualizations"],
        "category": "backend_data",
        "importance": 1.6,
    },
    "integrations": {
        "aliases": ["integration", "integrations", "third-party services"],
        "category": "backend_data",
        "importance": 1.9,
    },
    "git": {
        "aliases": ["git", "github", "gitlab"],
        "category": "professional",
        "importance": 1.3,
    },
    "linux": {
        "aliases": ["linux", "unix"],
        "category": "professional",
        "importance": 1.2,
    },
    "communication": {
        "aliases": ["communication", "communicator", "communicate", "presented", "explain", "technical report", "report", "documented"],
        "category": "professional",
        "importance": 1.4,
    },
    "degree": {
        "aliases": ["bachelor", "master", "computer science", "engineering", "mathematics", "math"],
        "category": "professional",
        "importance": 1.7,
    },
    "startup": {
        "aliases": ["startup", "founder", "founding", "y combinator", "yc"],
        "bridges": ["high agency", "initiative", "ownership"],
        "category": "professional",
        "importance": 1.5,
    },
    "biology": {
        "aliases": ["biology", "biologist", "genomics", "bioinformatics", "lab", "labs", "scientific"],
        "category": "professional",
        "importance": 0.8,
    },
}


def normalize_text(text: str) -> str:
    text = text.lower()

    replacements = {
        "ci/cd": "cicd",
        "ci-cd": "cicd",
        "node.js": "nodejs",
        "no-sql": "nosql",
        "no sql": "nosql",
        "full-stack": "fullstack",
        "full stack": "fullstack",
        "front-end": "frontend",
        "front end": "frontend",
        "back-end": "backend",
        "back end": "backend",
        "machine-learning": "machine learning",
        "deep-learning": "deep learning",
        "data-pipeline": "data pipeline",
        "data-pipelines": "data pipelines",
        "infrastructure-as-code": "infrastructure as code",
        "k8s": "kubernetes",
        "postgresql": "postgres",
        "google cloud": "gcp",
        "amazon web services": "aws",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("•", "\n• ")
    return text


def text_contains_alias(text: str, alias: str) -> bool:
    alias = normalize_text(alias)
    if any(ch in alias for ch in [" ", "+", ".", "/"]):
        return alias in text
    return re.search(rf"\b{re.escape(alias)}\b", text) is not None


def split_lines(text: str) -> List[str]:
    text = normalize_text(text)
    lines = [line.strip(" -\t") for line in text.splitlines()]
    return [line for line in lines if line.strip()]


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text)


def clean_tokens(tokens: List[str]) -> List[str]:
    cleaned = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in GENERIC_STOPWORDS:
            continue
        cleaned.append(token)
    return cleaned


def parse_resume_sections(resume_text: str) -> Dict[str, List[str]]:
    lines = split_lines(resume_text)
    sections: Dict[str, List[str]] = {
        "education": [],
        "skills": [],
        "projects": [],
        "experience": [],
        "leadership": [],
        "other": [],
    }

    current = "other"

    for line in lines:
        header = line.strip().lower()

        if header == "education":
            current = "education"
            continue
        if header in {"technical skills", "skills"}:
            current = "skills"
            continue
        if header == "projects":
            current = "projects"
            continue
        if header in {"academic experience", "experience"}:
            current = "experience"
            continue
        if header == "leadership":
            current = "leadership"
            continue
        if header == "languages":
            current = "other"
            continue

        sections[current].append(line)

    return sections


def parse_jd_sections(jd_text: str) -> List[Tuple[str, str]]:
    lines = split_lines(jd_text)
    sections: List[Tuple[str, List[str]]] = []
    current_name = "other"
    current_lines: List[str] = []

    for raw_line in lines:
        line = raw_line.strip().rstrip(":")
        key = SECTION_HEADERS.get(line)
        if key:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = key
            current_lines = []
        else:
            current_lines.append(raw_line)

    if current_lines:
        sections.append((current_name, current_lines))

    return [(name, "\n".join(lines_part)) for name, lines_part in sections]


def find_evidence_lines(resume_text: str, concept: str, limit: int = 2) -> List[str]:
    sections = parse_resume_sections(resume_text)
    spec = CONCEPTS[concept]
    aliases = list(spec.get("aliases", [])) + list(spec.get("bridges", []))
    hits: List[str] = []

    preferred_order = ["projects", "experience", "leadership", "skills", "education", "other"]

    for section in preferred_order:
        for line in sections.get(section, []):
            if any(text_contains_alias(line, alias) for alias in aliases):
                clean = re.sub(r"^\u2022\s*", "", line).strip()
                if clean not in hits:
                    hits.append(clean)
                if len(hits) >= limit:
                    return hits

    return hits[:limit]


def concept_present(text: str, concept: str) -> bool:
    spec = CONCEPTS[concept]
    for alias in spec.get("aliases", []):
        if text_contains_alias(text, alias):
            return True
    for alias in spec.get("bridges", []):
        if text_contains_alias(text, alias):
            return True
    return False


def compute_evidence_strength(resume_text: str, concept: str) -> float:
    sections = parse_resume_sections(resume_text)
    spec = CONCEPTS[concept]
    aliases = list(spec.get("aliases", [])) + list(spec.get("bridges", []))

    strong_verbs = {
        "built", "developed", "implemented", "designed", "created", "integrated",
        "deployed", "optimized", "authored", "engineered", "tuned", "managed",
    }

    strong_hits = 0
    medium_hits = 0
    weak_hits = 0

    def line_has_alias(line: str) -> bool:
        return any(text_contains_alias(line, alias) for alias in aliases)

    for section_name in ["projects", "experience", "leadership"]:
        for line in sections.get(section_name, []):
            if not line_has_alias(line):
                continue

            line_lower = line.lower()

            if concept == "monitoring":
                has_monitoring_word = any(x in line_lower for x in ["logging", "alerting", "tracing", "observability"])
                metrics_context = "metrics" in line_lower and any(x in line_lower for x in ["system", "service", "platform", "production", "monitoring"])
                if has_monitoring_word or metrics_context:
                    if any(x in line_lower for x in ["system", "service", "platform", "production", "pipeline"]):
                        strong_hits += 1
                    else:
                        medium_hits += 1
                continue

            if concept == "sql":
                if section_name in ["projects", "experience"]:
                    if any(x in line_lower for x in ["query", "queries", "schema", "database", "table", "storage", "join", "select", "insert"]):
                        strong_hits += 1
                    else:
                        medium_hits += 1
                else:
                    weak_hits += 1
                continue

            if concept == "database":
                if section_name in ["projects", "experience"]:
                    if any(x in line_lower for x in ["query", "queries", "schema", "data model", "storage", "table", "database"]):
                        strong_hits += 1
                    else:
                        medium_hits += 1
                else:
                    weak_hits += 1
                continue

            if any(v in line_lower for v in strong_verbs):
                strong_hits += 1
            else:
                medium_hits += 1

    for line in sections.get("skills", []):
        if line_has_alias(line):
            weak_hits += 1

    for section_name in ["education", "other"]:
        for line in sections.get(section_name, []):
            if line_has_alias(line):
                weak_hits += 1

    if strong_hits > 0:
        return 1.0
    if medium_hits > 0:
        return 0.7
    if weak_hits > 0:
        return 0.35
    return 0.0


def extract_jd_concepts(jd_text: str) -> Dict[str, Dict[str, object]]:
    sections = parse_jd_sections(jd_text)
    result: Dict[str, Dict[str, object]] = {}

    for section_name, section_text in sections:
        section_weight = SECTION_WEIGHTS.get(section_name, SECTION_WEIGHTS["other"])

        for concept, spec in CONCEPTS.items():
            if concept_present(section_text, concept):
                importance = float(spec["importance"])
                contribution = importance * section_weight

                if concept not in result:
                    result[concept] = {
                        "weight": 0.0,
                        "section_hits": set(),
                        "category": spec["category"],
                    }

                result[concept]["weight"] += contribution
                result[concept]["section_hits"].add(section_name)

    return result


def compute_resume_concepts(resume_text: str) -> Set[str]:
    found = set()
    for concept in CONCEPTS:
        if concept_present(resume_text, concept):
            found.add(concept)
    return found


def cosine_like_similarity(resume_text: str, jd_text: str) -> int:
    resume_tokens = clean_tokens(tokenize(resume_text))
    jd_tokens = clean_tokens(tokenize(jd_text))

    if not jd_tokens:
        return 0

    def tf(tokens: List[str]) -> Dict[str, float]:
        freq: Dict[str, float] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0.0) + 1.0
        size = max(len(tokens), 1)
        for token in list(freq.keys()):
            freq[token] /= size
        return freq

    rt = tf(resume_tokens)
    jt = tf(jd_tokens)
    vocab = set(rt) | set(jt)

    dot = sum(rt.get(v, 0.0) * jt.get(v, 0.0) for v in vocab)
    rn = math.sqrt(sum(v * v for v in rt.values()))
    jn = math.sqrt(sum(v * v for v in jt.values()))

    if rn == 0 or jn == 0:
        return 0

    score = int(round((dot / (rn * jn)) * 100))
    return max(0, min(score, 100))


def boosted_similarity(resume_text: str, jd_text: str) -> int:
    base = cosine_like_similarity(resume_text, jd_text)
    resume_concepts = compute_resume_concepts(resume_text)
    jd_concepts = set(extract_jd_concepts(jd_text).keys())

    overlap = len(resume_concepts & jd_concepts)
    bonus = min(overlap * 3, 24)

    return min(base + bonus, 100)


def compute_category_breakdown(
    jd_concepts: Dict[str, Dict[str, object]],
    resume_text: str
) -> Dict[str, int]:
    buckets = {
        "languages": 0.0,
        "backend_data": 0.0,
        "infra_devops": 0.0,
        "professional": 0.0,
    }
    matched_buckets = {k: 0.0 for k in buckets}

    for concept, meta in jd_concepts.items():
        category = meta["category"]
        weight = float(meta["weight"])
        buckets[category] += weight

        strength = compute_evidence_strength(resume_text, concept)
        matched_buckets[category] += weight * strength

    breakdown = {}
    for category, total in buckets.items():
        if total == 0:
            breakdown[category] = 0
        else:
            breakdown[category] = int(round((matched_buckets[category] / total) * 100))

    return breakdown


def score_match(
    jd_concepts: Dict[str, Dict[str, object]],
    resume_concepts: Set[str],
    resume_text: str
) -> Tuple[int, List[str], List[str]]:
    if not jd_concepts:
        return 0, [], []

    total = sum(float(meta["weight"]) for meta in jd_concepts.values())
    matched: List[str] = []
    missing: List[str] = []
    matched_weight = 0.0

    ordered = sorted(
        jd_concepts.items(),
        key=lambda item: (
            -float(item[1]["weight"]),
            CANONICAL_ORDER.index(item[0]) if item[0] in CANONICAL_ORDER else 999
        )
    )

    for concept, meta in ordered:
        strength = compute_evidence_strength(resume_text, concept)

        if strength > 0:
            matched.append(concept)
            matched_weight += float(meta["weight"]) * strength
        else:
            missing.append(concept)

    score = int(round((matched_weight / total) * 100)) if total else 0
    return score, matched, missing


def top_missing_by_priority(
    jd_concepts: Dict[str, Dict[str, object]],
    missing: List[str],
    limit: int = 6
) -> List[str]:
    ranked = sorted(missing, key=lambda c: -float(jd_concepts[c]["weight"]))
    return ranked[:limit]


def build_summary(
    overall_score: int,
    matched: List[str],
    missing: List[str],
    category_breakdown: Dict[str, int],
    semantic_score: int
) -> str:
    if overall_score >= 80:
        level = "strong"
    elif overall_score >= 65:
        level = "solid"
    elif overall_score >= 45:
        level = "partial"
    else:
        level = "limited"

    matched_preview = ", ".join(matched[:5]) if matched else "none yet"
    missing_preview = ", ".join(missing[:5]) if missing else "none"
    best_bucket = max(category_breakdown.items(), key=lambda x: x[1])[0].replace("_", " ")

    return (
        f"This resume shows {level} alignment with the role. "
        f"Best coverage is in {best_bucket}. "
        f"Matched areas include {matched_preview}. "
        f"The biggest gaps are {missing_preview}. "
        f"Similarity is {semantic_score}/100, while the weighted score distinguishes between "
        f"surface-level mentions and evidence-backed experience."
    )


def build_suggestions(missing_priority: List[str], resume_text: str) -> List[str]:
    suggestions = []
    missing_set = set(missing_priority)

    if missing_priority:
        suggestions.append(
            "Target the highest-value gaps first: " + ", ".join(missing_priority[:5]) + "."
        )

    if {"api", "backend"} & missing_set:
        suggestions.append(
            "Rewrite one project as backend work: mention endpoints, request/response flow, service logic, or integrations instead of only 'pipeline' or 'system'."
        )

    if {"sql", "database", "nosql"} & missing_set:
        suggestions.append(
            "Add explicit database evidence. Even class or project usage counts if you name SQL, schema design, queries, or storage choices."
        )

    if {"cloud", "aws", "gcp", "docker", "kubernetes", "cicd", "monitoring", "terraform"} & missing_set:
        suggestions.append(
            "Your current resume under-signals infra. Add deployment, containerization, Linux environment setup, GitHub Actions, logging, or any cloud experimentation if you have it."
        )

    if "startup" in missing_set:
        suggestions.append(
            "For startup roles, emphasize ownership: shipped independently, moved fast with ambiguity, iterated quickly, or worked without heavy supervision."
        )

    if "communication" in missing_set and "technical report" in normalize_text(resume_text):
        suggestions.append(
            "You already have communication evidence. Surface it harder by mentioning presenting results, explaining tradeoffs, or documenting decisions."
        )

    if len(suggestions) < 5:
        suggestions.append(
            "Mirror the JD's exact nouns in your bullets. Recruiters and lightweight ATS scoring systems reward wording alignment."
        )

    return suggestions[:5]


def collect_bullet_lines(resume_text: str) -> List[str]:
    sections = parse_resume_sections(resume_text)
    bullets: List[str] = []
    for section_name in ["projects", "experience", "leadership"]:
        for line in sections.get(section_name, []):
            if line.strip().startswith("•") or line.strip().startswith("-"):
                bullets.append(re.sub(r"^[•\-]\s*", "", line).strip())
    return bullets


def rewrite_bullet(line: str, jd_concepts: Set[str]) -> str:
    line_lower = line.lower()
    rewritten = line

    if "pipeline" in line_lower and "api" in jd_concepts:
        rewritten += " Exposed functionality via RESTful API endpoints for data access."

    if ("data" in line_lower or "pipeline" in line_lower) and "database" in jd_concepts:
        rewritten += " Integrated with a structured database for data storage and query handling."

    if "system" in line_lower and "backend" in jd_concepts:
        rewritten = rewritten.replace(
            "system",
            "backend system with defined data flow and service architecture"
        )

    if "cloud" in jd_concepts and "built" in line_lower:
        rewritten += " Designed with cloud deployment considerations (AWS/GCP)."

    return rewritten


@app.get("/")
def root():
    return {"message": "Resume Matcher API v4 is running."}


@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    resume_text = normalize_text(data.resume)
    jd_text = normalize_text(data.job_description)

    jd_concepts = extract_jd_concepts(jd_text)
    resume_concepts = compute_resume_concepts(resume_text)

    keyword_score, matched, missing = score_match(jd_concepts, resume_concepts, resume_text)
    semantic_score = boosted_similarity(resume_text, jd_text)

    category_breakdown = compute_category_breakdown(jd_concepts, resume_text)
    missing_priority = top_missing_by_priority(jd_concepts, missing, limit=6)

    overall_score = int(round(0.82 * keyword_score + 0.18 * semantic_score))

    evidence = {
        concept: find_evidence_lines(resume_text, concept)
        for concept in matched[:8]
    }

    summary = build_summary(overall_score, matched, missing_priority, category_breakdown, semantic_score)
    suggestions = build_suggestions(missing_priority, resume_text)

    return {
        "score": overall_score,
        "keyword_score": keyword_score,
        "semantic_score": semantic_score,
        "matched_keywords": matched[:12],
        "missing_keywords": missing_priority[:12],
        "summary": summary,
        "suggestions": suggestions,
        "breakdown": {
            "languages": category_breakdown["languages"],
            "backend_data": category_breakdown["backend_data"],
            "infra_devops": category_breakdown["infra_devops"],
            "professional": category_breakdown["professional"],
        },
        "evidence": evidence,
        "all_detected_jd_concepts": [
            {
                "concept": concept,
                "weight": round(float(meta["weight"]), 2),
                "category": meta["category"],
                "sections": sorted(list(meta["section_hits"])),
                "evidence_strength": compute_evidence_strength(resume_text, concept),
            }
            for concept, meta in sorted(jd_concepts.items(), key=lambda x: -float(x[1]["weight"]))
        ],
    }


@app.post("/rewrite")
def rewrite(data: RewriteRequest):
    resume_text = normalize_text(data.resume)
    jd_text = normalize_text(data.job_description)

    jd_concepts = set(extract_jd_concepts(jd_text).keys())
    bullets = collect_bullet_lines(data.resume)

    rewritten_bullets = []
    for bullet in bullets:
        rewritten = rewrite_bullet(bullet, jd_concepts)
        if rewritten != bullet:
            rewritten_bullets.append({
                "original": bullet,
                "rewritten": rewritten,
            })

    return {
        "count": len(rewritten_bullets),
        "rewritten_bullets": rewritten_bullets[:8],
        "detected_jd_concepts": sorted(list(jd_concepts)),
    }
