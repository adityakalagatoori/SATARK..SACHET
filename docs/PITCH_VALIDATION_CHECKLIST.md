# SACHET Pitch Validation Checklist

Derived from official PSB Hackathon Series 2025 winner synopses published by DFS
(financialservices.gov.in/psb-hackathon). Five winning pitches were analyzed directly
from source PDFs, not secondary reporting:

| Bank | Team | Solution | Problem domain |
|---|---|---|---|
| Bank of India (our hackathon, 2025 edition) | Jigyasa | SuRaksha | Mobile banking impersonation / password-less auth |
| Bank of India | Doom n Gloom | RISKON | Credit risk scoring, underbanked population |
| Bank of Baroda | 200_OK | Intelligent API Security Fabric | Zero-trust API security |
| Punjab National Bank | PwnedRaccoons | HBRPS | Host-based ransomware defense |
| State Bank of India | Pathfinders | FINDMARK | Loan defaulter detection/tracking |
| Canara Bank | Finavat (1st) / Jigyasa (3rd) | — / SuRaksha variant | Mobile banking security, PII protection |

Use this file to check the SACHET deck against what juries in this exact series have
actually rewarded — not generic pitch advice.

---

## 1. Structural shape (mirrors every official winner synopsis)

Every synopsis DFS published follows this exact shape. A deck that visibly hits each
beat reads as "native" to how these juries already evaluate:

- [ ] Opens by stating the problem's relevance to banking/FinTech in plain terms — no jargon before context
- [ ] Names the solution with a real name (not "our system" or "the model")
- [ ] Describes the solution as a plain-language paragraph before any architecture
- [ ] Includes a section on real-world deployment / bank partnership plan (not left implicit)
- [ ] Closes with outcomes/learnings/future plans, stated as concrete next steps
- [ ] States numbers/stats in a dedicated "notable points" beat, not buried mid-paragraph

## 2. Judging criteria confirmed across banks — score against each explicitly

Different banks stated different explicit axes. Assume all of these are live for BOI:

- [ ] **Technical feasibility** — is a working, non-hand-wavy build demonstrable, not just described?
- [ ] **Business potential** — is there a clear "how BOI would actually deploy/monetize this" statement?
- [ ] **Scalability** — is there a specific, credible claim (not "it scales"), ideally with a number or load scenario? (SBI mentors explicitly penalized vague scalability claims — "academic proof-of-concepts often underestimate real-world scalability")
- [ ] **Uniqueness** — is there one clearly stated thing no other likely competitor has?
- [ ] **Accuracy** — are real, honestly-labeled metrics shown (not inflated, not vague)?
- [ ] **Ethical considerations** — is data handling (customer transaction data, PII) addressed proactively, before a judge has to ask?
- [ ] **Real-world implementation potential** — is there a believable path from prototype to production, not just "future work"?

## 3. Rhetorical devices actually used by winners

- [ ] **One quotable thesis line** stated early, that the rest of the pitch keeps returning to (SBI: *"In a world of imperfect data, true innovation lies in creating clarity from chaos."*)
- [ ] **One named, memorable sub-component** with a hard, specific (not rounded) number attached (SBI's FINDMARK: "90% accuracy within a 10-meter radius"; PNB's HBRPS: "under 100MB")
- [ ] **One standout, demoable feature** stated as a single sharp line, not folded into a feature list (Jigyasa's SuRaksha: "triple-tap emergency logout")
- [ ] **One vivid technical metaphor or visual centerpiece** the whole room remembers (BoB's "API Health Orb" turning red; RISKON's "creditworthiness as a physical particle responding to a force field")
- [ ] **Grounded urgency via specific, dated real incidents** — not vague industry stats (PNB's ransomware timeline table: WannaCry 2017 → C-Edge Technologies hit 300 Indian banks 2024 → DBS/BoC 2025)
- [ ] **National-mission framing** present somewhere (Digital India / Atmanirbhar Bharat / indigenous self-reliance) — appeared in BoB, PNB, and Canara Bank leadership quotes independently, so it's an expected, not optional, beat
- [ ] **Explicit phased, bank-specific rollout plan** (Phase 1/2/3), not a generic roadmap — ideally naming a BOI-specific pilot surface (e.g. a specific channel/product) the way BoB proposed piloting on Baroda's own payment APIs

## 4. Naming & branding

- [ ] Solution has a real name with an evocative meaning, ideally Hindi-rooted (SuRaksha, RISKON, SU₹क्षा wordmark with rupee symbol embedded) — SACHET / SATARK (सतर्क = "Vigilant") already fits this pattern, keep it prominent, don't bury the meaning
- [ ] If there's a visual/logo, check for a small clever visual pun opportunity (Canara's ₹ embedded in "Suraksha") — not required, but a differentiator when present

## 5. Anti-patterns / red flags seen or implied as penalized

- [ ] No claim of a capability with no number behind it ("AI-powered," "highly scalable," "robust") — every one of these winners attached a number or named mechanism instead
- [ ] No scalability claim without a load scenario or real benchmark — mentors explicitly flagged inflated scale claims as a weakness during SBI mentorship
- [ ] No ethics/data-handling section skipped or left to Q&A — SBI's winner stated it proactively in the main writeup
- [ ] No generic "future work" — every winner named a concrete next step (a specific event, a specific bank integration, a specific expansion)
- [ ] No architecture-first opening — every synopsis explains relevance and the plain-language solution before any technical section

## 6. BOI-specific notes (this is literally last year's edition of our hackathon)

- Our jury pool precedent: IDRBT, IIT faculty, **McKinsey & Deloitte**, plus BOI's own IT/Risk/Data Analytics teams — consulting-caliber scrutiny on business potential and scalability, not just a coding-demo bar.
- BOI scored every submission **twice** for objectivity — so the writeup/deck must read cleanly on a second, cold pass, not just deliver well live.
- Prior BOI PS2 winner (SuRaksha) beat the room on **frictionless UX** framing over raw technical depth — for SACHET, lead with what changes for a BOI ops analyst/customer, not just model internals.

---

## 7. Complete Slide-by-Slide Content Checklist (minute → major, nothing skipped)

Master checklist for the actual PPT build. Work top to bottom; every unchecked box is
either a missing slide or a missing detail on an existing one.

### 7.1 Title / cover slide
- [ ] Solution name (SACHET) prominent, larger than team name
- [ ] Meaning of the name shown, not just the name (सतर्क / "Vigilant") — a name nobody explains is a wasted hook
- [ ] Team name (SATARK)
- [ ] Full hackathon name, problem statement number, bank, and academic partner ("PSB Cybersecurity, Fraud & AI Hackathon 2026 | Problem Statement 2 | Bank of India × IIT Hyderabad")
- [ ] Date of presentation
- [ ] All team members' full names (spelled correctly) and college/institute
- [ ] A one-line strapline under the name (the thesis-quote equivalent) — not left for slide 2

### 7.2 Agenda / roadmap slide
- [ ] Short outline of what's coming (Problem → Solution → Demo → Results → Business Plan → Roadmap) so judges know the shape and don't wonder "when do we get to X"
- [ ] Time budget implied or stated if the slot is tight

### 7.3 Problem statement slide
- [ ] Restate the official problem statement in the bank's own wording (don't paraphrase into something looser)
- [ ] Explain why it matters in one plain-language paragraph before any numbers
- [ ] At least one grounded, dated, named real-world incident or statistic (not a vague industry-wide claim) — e.g., a specific mule-account fraud case or RBI/NPCI figure with a year attached
- [ ] Explicitly name who is harmed today (customer, bank, regulator) and how

### 7.4 Solution overview / one-liner slide
- [ ] One sentence describing what SACHET does, sayable in under 10 seconds, no jargon
- [ ] The one memorable visual/metaphor for the whole system (our "API Health Orb" equivalent) — likely the AUTO-HOLD dashboard reacting live, or the 4-tier alert visualization
- [ ] Explicit statement of what's different from a bank's existing rule-based fraud system

### 7.5 Architecture / how it works
- [ ] One diagram showing the full pipeline end-to-end (ingestion → feature store → ensemble scoring → governance/kill-switch → customer disclosure) — one diagram, not five
- [ ] Each named component labeled with a plain-language purpose, not just a box and an acronym
- [ ] Data sources listed (what feeds the model) and explicitly which are real vs. mocked/simulated for the demo
- [ ] Latency / throughput claim stated as a number, not "real-time" alone
- [ ] Explicit callout of what's production-ready vs. prototype-only — the honesty move that reads as credibility, not weakness

### 7.6 Model / technical proof slide(s)
- [ ] Real accuracy/precision/recall/AUC numbers, labeled exactly what they measure (not just "accuracy: 99%" with no context)
- [ ] Confusion matrix or equivalent — show false positive/negative tradeoff honestly
- [ ] Which models are in the ensemble and why each was chosen (tie back to "Why these 5 models" reasoning already in submission.html)
- [ ] Ablation or component-vs-ensemble comparison (already have this in submission.html — reuse the strongest single chart, not all of them)
- [ ] Explicit fairness/bias check result stated, not just mentioned as having been done
- [ ] A named, hard number for the standout technical claim (our "90% in 10m radius" equivalent) — pick the single most defensible number, not the biggest-sounding one

### 7.7 Live demo section
- [ ] Demo script decided in advance: exact sequence of clicks/inputs, not improvised
- [ ] A pre-recorded backup video of the same demo in case live fails (Wi-Fi, laptop, projector risk)
- [ ] Demo shows the kill-switch / human-override path, not just a score popping up — governance-in-action is the differentiator
- [ ] Demo includes at least one "boring" transaction that correctly does NOT trigger an alert (shows precision, not just recall)
- [ ] Screen/font size checked to be readable from the back of the room, not just on a laptop

### 7.8 Governance, ethics & compliance slide
- [ ] Explicit statement on customer data handling (what's collected, retention, DPDP Act / RBI data-localization alignment) — stated proactively, not left for Q&A
- [ ] Explanation of the customer disclosure notice — how a customer is told when auto-held, in plain language
- [ ] Human-override / kill-switch explained as a deliberate safety control, not an afterthought
- [ ] Explicit note on fairness auditing across demographic/account-type slices

### 7.9 Business viability & deployment plan
- [ ] Explicit "how BOI would actually deploy this" statement — which system it plugs into (Finacle, core banking, UPI switch), not just "the API"
- [ ] Named phased rollout: Phase 1 / Phase 2 / Phase 3, each with a concrete scope (e.g., pilot on one channel → expand to cross-bank signal via DPIP/I4C → full production)
- [ ] Cost/effort honesty: what BOI would need to provide (compute, data access, integration effort) — not glossed over
- [ ] A believable ROI or harm-reduction estimate, sourced from a real number (RBI mule-account loss figures, etc.), not invented

### 7.10 Uniqueness / competitive differentiation slide
- [ ] One slide stating explicitly what no other likely competing team is doing (the cross-bank signal / DPIP-I4C integration angle is a strong, distinct candidate)
- [ ] Comparison against status-quo rule-based systems, framed on a metric (false positive rate, detection lag), not adjectives

### 7.11 Limitations / honesty slide
- [ ] At least one explicit, named limitation stated before a judge can find it (e.g., the schema-mapping gap between rolling features and the trained ensemble's exact columns, if still true at presentation time)
- [ ] Framed as "what we'd solve next," not as an apology

### 7.12 Roadmap / future scope slide
- [ ] Concrete next step named — not generic "future work" (a specific integration target, a specific pilot, a specific event)
- [ ] Tied explicitly to a national-mission framing somewhere on this slide or the closing slide (Digital India / financial inclusion / Atmanirbhar Bharat) — every analyzed winner had this beat

### 7.13 Team / credibility slide
- [ ] Team name and members again (some judges only see this slide if they missed the title)
- [ ] One line per member on role/contribution, not just names
- [ ] Institute affiliation

### 7.14 Closing slide
- [ ] One-line callback to the opening thesis/strapline — bookend the pitch
- [ ] Clear call to action (what you're asking the bank for: a pilot, data access, mentorship)
- [ ] Contact info / QR code for follow-up

### 7.15 Backup / appendix slides (not shown unless asked, but ready)
- [ ] Full metrics tables (all models, all folds) for a technical judge who asks
- [ ] Full architecture diagram with every component labeled
- [ ] Data schema / feature list for anyone who asks "what exactly feeds this"
- [ ] References / sources for every statistic used in the main deck
- [ ] Extra Q&A-anticipation slide: likely tough questions and one-line answers ready (e.g., "how do you handle adversarial/laundered behavior," "what's your false positive cost to the bank")

### 7.16 Design & formatting minutiae
- [ ] Consistent color palette across every slide (tie to BOI's brand colors if possible, not clashing)
- [ ] Consistent font, consistent slide numbering, consistent header style
- [ ] No walls of text — every slide should be readable in under 10 seconds at a glance
- [ ] Every chart has a legend and axis labels, no unlabeled numbers
- [ ] Spelling check on every proper noun: "Bank of India," "IIT Hyderabad," bank officials' names if quoted, model names
- [ ] Page/slide numbers present ("Page X of Y") for judges taking notes
- [ ] High-contrast, readable-from-distance font sizes (test by standing at the back of a room)

### 7.17 Delivery & logistics minutiae
- [ ] Full run-through rehearsed against the actual time limit, not estimated
- [ ] Laptop/adapter/clicker checked and a backup (USB copy, PDF export) carried
- [ ] Offline fallback for the live demo confirmed (no dependency on live internet if avoidable)
- [ ] Clear division of who speaks which section, rehearsed handoffs
- [ ] Anticipated Q&A questions assigned to specific team members in advance
- [ ] Printed one-pager or leave-behind summary for judges, if allowed
- [ ] Version/date stamp on the deck so an old draft is never accidentally presented
