# AgentFlow AI &mdash; Autonomous Multi-Agent Research Platform

An evidence-based multi-agent research-to-action platform built with Python, Streamlit, Gemini API, asyncio, SQLite, ReportLab, Pandas, and Pydantic. Designed specifically for complex Indian healthcare procurement, compliance auditing, and workflow automation.

---

## 🎯 Benchmark Problem Statements

1. **Primary Benchmark**: Evaluate patient appointment-reminder software for an Indian hospital operations manager and generate an evidence-backed shortlist across 7 Indian healthcare standards.
2. **Secondary Benchmark**: Compare telemedicine platforms tailored for rural clinics in India (demonstrating dynamic DAG adaptation to task requirements).

---

## 🏗️ Architecture & Agent Roles

AgentFlow AI is structured into four specialized, loosely-coupled modules:

```
                  ┌──────────────────────┐
                  │      User Task       │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │    Manager Agent     │ (M1: DAG decomposition, cycle checks)
                  └──────────┬───────────┘
                             │ Plan DAG
                             ▼
              ┌──────────────────────────────┐
              │      Async Orchestrator      │ (M1: asyncio.gather, failure isolation)
              └──────────────┬───────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
    ┌────────▼────────┐             ┌────────▼────────┐
    │   Research A    │             │   Research B    │ (M2: Tools, DDG search, trafilatura)
    │ (Features/DLT)  │             │ (Pricing/INR)   │
    └────────┬────────┘             └────────┬────────┘
             │                               │
             └───────────────┬───────────────┘
                             │ Claims
                    ┌────────▼────────┐
                    │    Analysis     │ (M3: Conflict detection, India scoring)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Verifier     │ (M3: Verbatim snippet fuzzy matching)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Reporter / App  │ (M4: PDF dossier, Streamlit UI, HITL)
                    └─────────────────┘
```

### Module Breakdown:
1. **M1 (Lead / Orchestrator)**:
   - `schemas.py`: Pydantic data contracts (`Claim`, `Plan`, `PlanStep`, `Conflict`, `VerificationResult`, `Event`, `Analysis`, `Report`).
   - `db.py`: SQLite event logging, task history, and human-in-the-loop approval tables.
   - `llm.py`: Production-grade Gemini API wrapper with exponential backoff, jitter, validation retry, and semaphore concurrency limiting.
   - `agents/manager.py`: Lead planner decomposing instructions into dependency DAGs with Kahn's algorithm cycle detection.
   - `orchestrator.py`: Dynamic executor running independent steps in parallel waves (`asyncio.gather`), per-step timeouts, and targeted retry loops.
2. **M2 (Research & Tools)**:
   - `tools/fetch.py`: Async web fetcher using `httpx` + `trafilatura` with typed `FetchError` (timeout, paywall, 403, JS-only).
   - `tools/cache.py`: SQLite caching for search queries and page fetches with optional TTL.
   - `tools/search.py`: Async search using `duckduckgo-search` with relevance filtering (dropping dictionary/encyclopedia domains) and fallback backoff.
   - `agents/research.py`: Autonomous dual-track research (`run_research_a`, `run_research_b`) extracting verbatim evidence snippets.
3. **M3 (Analysis & Verification)**:
   - `agents/analysis.py`: Groups claims by vendor and attribute, performs deterministic conflict detection, evaluates 7 India criteria, and builds feature/pricing matrices with explicit `"unknown"` cells.
   - `agents/verifier.py`: Deterministic-first evidence verification via local re-fetch, whitespace normalization, rapidfuzz/difflib fuzzy snippet matching, and LLM claim support checks.
4. **M4 (Reporting, UI, and HITL Action)**:
   - `agents/reporter.py`: Synthesizes executive findings, generates procurement dossiers, and drafts personalized vendor RFQ quote emails.
   - `reports/generate_pdf.py`: Professional PDF dossier generation using ReportLab.
   - `app.py`: Full-featured Streamlit dashboard featuring live workflow execution, matrix view, evidence audits, and human approval gates.

---

## 🇮🇳 The 7 Indian Healthcare Standards

All software vendors are quantitatively scored against seven non-negotiable Indian clinical and regulatory criteria:
1. **WhatsApp Business API**: Conversational automated reminders, interactive confirmations, and template opt-ins.
2. **TRAI / DLT SMS Compliance**: Pre-approved header whitelisting for transactional alerts (NDNC registry compliant).
3. **Regional Language Support**: Multilingual outreach (Hindi, Tamil, Telugu, Kannada, etc.) to minimize OPD patient no-shows.
4. **ABDM / ABHA Integration**: M1, M2, and M3 milestone interoperability with India's Ayushman Bharat Digital Mission.
5. **DPDP Act 2023 Compliance**: Strict patient consent mechanisms and in-country healthcare data residency.
6. **HIS / EMR Interoperability**: Direct HL7, FHIR, and REST connectors for seamless clinical hospital integration.
7. **Transparent INR Pricing**: Clear, recurring subscription tiers in Indian Rupees without foreign currency conversion markups.

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/Vadirajsangam06/Pack-Man.git
cd Pack-Man
pip install -r requirements.txt
```

### 2. Environment Setup
Copy the example environment configuration:
```bash
cp .env.example .env
# Edit .env and supply GEMINI_API_KEY
```

### 3. Run the Full Test Suite
Execute all 68+ unit, integration, and contract tests:
```bash
pytest -v
```

### 4. Launch the Streamlit Dashboard
Start the interactive application:
```bash
streamlit run app.py
```

---

## 🛡️ Human-in-the-Loop (HITL) Safety Architecture

AgentFlow AI strictly enforces that autonomous research agents **never** communicate with external vendors or trigger transactions without explicit human review:
- Outbound RFQs are placed in a `pending` state inside the SQLite `approvals` table.
- Operations leaders inspect the proposed RFP email in the dashboard.
- Clicking **"Approve & Dispatch"** records the operator's decision and dispatches the action with full audit provenance.
