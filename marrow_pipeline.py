"""
marrow_pipeline.py - Full Marrow PYQ PDF pipeline
Provides all functions required by extract.py
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ══════════════════════════════════════════════════════════════════
# PAGE TYPES
# ══════════════════════════════════════════════════════════════════

class PageType:
    FRONT       = "front"
    CATALOG     = "catalog"
    SCHEMA      = "schema"
    MCQ         = "mcq"
    NOISE       = "noise"
    UNKNOWN     = "unknown"


# ══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════

@dataclass
class CatalogEntry:
    topic      : str
    page_number: int


@dataclass
class SchemaSet:
    title      : str
    mcq_count  : int
    first_page : int
    topics     : List[str] = field(default_factory=list)


@dataclass
class MCQ:
    question_id      : str       = ""
    pdf_stem         : str       = ""
    subject          : str       = ""
    topic            : str       = ""
    pearl_id         : str       = ""
    page             : int       = 0
    question         : str       = ""
    question_html    : str       = ""
    question_image   : str       = ""   # rendered PNG crop of question+options
    options          : Dict[str,str] = field(default_factory=dict)
    answer           : str       = ""
    explanation      : str       = ""
    explanation_html : str       = ""
    explanation_image: str       = ""   # rendered PNG crop of explanation
    images           : List[str] = field(default_factory=list)
    topic_tags       : List[str] = field(default_factory=list)
    schema_topics    : List[str] = field(default_factory=list)
    flags            : List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# SUBJECT MAP
# ══════════════════════════════════════════════════════════════════

SUBJECT_KEYWORDS: Dict[str, List[str]] = {
    "anatomy"      : ["anat","anatomy","embryo","histol","gross"],
    "physiology"   : ["physio","physiolog"],
    "biochemistry" : ["biochem","bioch","molecular","metabol"],
    "pathology"    : ["path","histopath","cytol","neoplasm"],
    "pharmacology" : ["pharma","drug","receptor","toxicol"],
    "microbiology" : ["micro","bacterio","virol","mycol","parasit",
                      "immunol","serol"],
    "medicine"     : ["medicine","internal","clinical"],
    "surgery"      : ["surgery","surg","surgical"],
    "psm"          : ["psm","community","preventive","social",
                      "epidem","public health","biostat"],
    "ophthalmology": ["ophthalm","eye","ocul","vision"],
    "ent"          : ["ent","otol","rhinol","laryn","ear","nose","throat"],
    "pediatrics"   : ["pediatr","paediatr","neonat","child"],
    "orthopedics"  : ["ortho","fracture","spine","bone","joint"],
    "radiology"    : ["radio","imaging","xray","x-ray","mri","ct"],
    "dermatology"  : ["dermat","skin"],
    "psychiatry"   : ["psychiat","mental","behav","dsm"],
    "anaesthesia"  : ["anaesth","anesthes","sedation","icu"],
    "obgyn"        : ["obstet","gynecol","gynaecol","obgyn","maternal"],
    "forensic"     : ["forensic","legal","medicolegal"],
}


def detect_subject(text: str) -> str:
    """Guess subject from text (filename, page text, etc)."""
    t      = text.lower()
    scores : Dict[str, int] = {}
    for subj, kws in SUBJECT_KEYWORDS.items():
        score = sum(t.count(kw) for kw in kws)
        if score:
            scores[subj] = score
    return max(scores, key=scores.get) if scores else "unknown"


def detect_subject_from_pdf(pdf_path: str, questions: list) -> str:
    """
    Detect subject from PDF filename first,
    then fall back to question text.
    """
    from pathlib import Path
    stem    = Path(pdf_path).stem.lower()
    subject = detect_subject(stem)
    if subject != "unknown":
        return subject
    # Try from questions
    combined = " ".join(
        q.get("question", "") for q in questions[:20]
    )
    return detect_subject(combined)


# ══════════════════════════════════════════════════════════════════
# PATTERNS
# ══════════════════════════════════════════════════════════════════

RE_ANS = re.compile(
    r"""
    (?:ans(?:wer)?|correct\s*ans(?:wer)?|key)
    \s*[:\-]?\s*
    \(?([A-Da-d])\)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

RE_OPT = re.compile(
    r"""
    (?:^|\n)\s*
    (?:\(\s*)?([A-Da-d])(?:\s*\)|\s*[\.\-\:])\s*
    (.+?)
    (?=
        (?:\n\s*(?:\(\s*)?[A-Da-d](?:\s*\)|\s*[\.\-\:]))  |
        (?:\n\s*(?:ans|answer|correct|key)\s*[:\-])        |
        $
    )
    """,
    re.VERBOSE | re.DOTALL | re.IGNORECASE,
)

RE_PEARL = re.compile(
    r"\b([A-Z]{2,6})\s*[-_]\s*(\d{1,4})\b"
)

RE_NOISE = re.compile(
    r"""
    ©\s*[Mm]arrow       |
    marrow\.in          |
    \bSOLVE\b           |
    \bNEXT\b            |
    \bBOOKMARK\b        |
    \bCOMPLETE\b        |
    \bPREMIUM\b         |
    \b[a-f0-9]{20,}\b   |
    \[\s*\d{1,3}%\s*\]  |
    Page\s+\d+\s+of\s+\d+
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

RE_CATALOG_LINE = re.compile(
    r"^(.+?)\s*[\.\-]+\s*(\d+)\s*$",
    re.MULTILINE,
)

RE_SCHEMA_TITLE = re.compile(
    r"(?:chapter|unit|section|topic)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)

RE_QNUM = re.compile(
    r"(?:^|\n)\s*(?:Q\.?\s*\d+|\d+\s*\.|\(\s*\d+\s*\))\s*",
    re.MULTILINE,
)


# ══════════════════════════════════════════════════════════════════
# PAGE CLASSIFIER
# ══════════════════════════════════════════════════════════════════

def classify_page(page) -> str:
    """
    Classify a pdfplumber page object.
    Returns a PageType constant string.
    """
    try:
        text = page.extract_text() or ""
    except Exception:
        return PageType.NOISE

    if not text.strip():
        return PageType.NOISE

    text_low = text.lower()

    # Front/cover page — very short or has cover keywords
    if len(text.strip()) < 100:
        return PageType.FRONT

    if any(kw in text_low for kw in [
        "table of content", "contents", "index", "syllabus"
    ]):
        # Only if it has page numbers (catalog pattern)
        if RE_CATALOG_LINE.search(text):
            return PageType.CATALOG
        return PageType.FRONT

    # Schema / chapter header
    if RE_SCHEMA_TITLE.search(text) and len(text.strip()) < 400:
        return PageType.SCHEMA

    # Noise — mostly watermarks
    clean = RE_NOISE.sub("", text).strip()
    if len(clean) < 50:
        return PageType.NOISE

    # Default — treat as MCQ page
    return PageType.MCQ


# ══════════════════════════════════════════════════════════════════
# CATALOG EXTRACTOR
# ══════════════════════════════════════════════════════════════════

def extract_catalog(page) -> List[CatalogEntry]:
    """
    Extract topic → page_number pairs from a catalog/TOC page.
    Returns list of CatalogEntry.
    """
    entries: List[CatalogEntry] = []
    try:
        text = page.extract_text() or ""
    except Exception:
        return entries

    for m in RE_CATALOG_LINE.finditer(text):
        topic   = m.group(1).strip()
        pg_num  = int(m.group(2))
        if topic and pg_num:
            entries.append(CatalogEntry(
                topic       = topic,
                page_number = pg_num,
            ))
    return entries


# ══════════════════════════════════════════════════════════════════
# SCHEMA EXTRACTOR
# ══════════════════════════════════════════════════════════════════

def extract_schema(page, page_idx: int) -> SchemaSet:
    """
    Extract schema/chapter metadata from a schema page.
    Returns SchemaSet.
    """
    try:
        text = page.extract_text() or ""
    except Exception:
        text = ""

    title    = ""
    topics   : List[str] = []
    mcq_count = 0

    # Title
    m = RE_SCHEMA_TITLE.search(text)
    if m:
        title = m.group(1).strip()

    # MCQ count
    count_m = re.search(
        r"(\d+)\s*(?:mcq|question|q\.?s?)",
        text, re.IGNORECASE
    )
    if count_m:
        mcq_count = int(count_m.group(1))

    # Topics — bullet lines
    for line in text.splitlines():
        line = line.strip()
        if (line
                and len(line) > 3
                and not re.match(r"^\d+$", line)
                and line.lower() not in ("chapter","unit","section")):
            topics.append(line)

    return SchemaSet(
        title      = title or text[:60].strip(),
        mcq_count  = mcq_count,
        first_page = page_idx,
        topics     = topics[:10],
    )


# ══════════════════════════════════════════════════════════════════
# PEARL METADATA EXTRACTOR
# ══════════════════════════════════════════════════════════════════

def extract_pearl_metadata(page) -> Optional[Dict[str, str]]:
    """
    Extract pearl ID, subject, topic from page header/footer.
    Returns dict or None.
    """
    try:
        text = page.extract_text() or ""
    except Exception:
        return None

    # Pearl ID pattern e.g. MICR-042
    pm = RE_PEARL.search(text[:200])
    if not pm:
        return None

    pearl_id = f"{pm.group(1)}-{pm.group(2)}"
    subject  = detect_subject(pm.group(1))

    # Topic — line after pearl id
    lines = text[:200].splitlines()
    topic = ""
    for i, line in enumerate(lines):
        if pearl_id in line and i + 1 < len(lines):
            topic = lines[i + 1].strip()
            break

    return {
        "pearl_id": pearl_id,
        "subject" : subject,
        "topic"   : topic,
    }


# ══════════════════════════════════════════════════════════════════
# TEXT CLEANER
# ══════════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    if not text:
        return ""
    text = RE_NOISE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ",    text)
    return text.strip()


def _to_html(text: str) -> str:
    if not text:
        return ""
    import html as html_mod
    t = html_mod.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\b_(.+?)_\b",   r"<i>\1</i>", t)
    paras = re.split(r"\n{2,}", t)
    return "".join(
        f"<p>{p.replace(chr(10), ' ')}</p>"
        for p in paras if p.strip()
    )


# ══════════════════════════════════════════════════════════════════
# OPTION PARSER
# ══════════════════════════════════════════════════════════════════

def _parse_options(text: str) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for m in RE_OPT.finditer(text):
        letter = m.group(1).upper()
        val    = re.sub(r"\s+", " ", m.group(2)).strip()
        opts[letter] = val
    return opts


# ══════════════════════════════════════════════════════════════════
# MCQ PARSER
# ══════════════════════════════════════════════════════════════════

def parse_mcq(
    stream       : List[Dict],
    answer       : Optional[str],
    schema_topics: List[str],
    page_idx     : int,
    q_id         : str,
) -> MCQ:
    """
    Parse a stream of {type, plain/html/tag/file} dicts into an MCQ.

    stream items:
      {"type":"text",  "plain":"...", "html":"..."}
      {"type":"image", "tag":"...",   "file":"..."}
    """
    # Collect all text
    text_parts : List[str] = []
    html_parts : List[str] = []
    images     : List[str] = []

    for item in stream:
        if item.get("type") == "image":
            images.append(item.get("file", ""))
        else:
            text_parts.append(item.get("plain", ""))
            html_parts.append(item.get("html",  ""))

    full_text = "\n".join(text_parts)
    full_html = "".join(html_parts)
    full_text = _clean(full_text)

    # Find answer
    ans = answer
    if not ans:
        m = RE_ANS.search(full_text)
        if m:
            ans = m.group(1).upper()

    # Split stem / options
    opt_start = re.search(
        r"""(?:^|\n)\s*(?:\(\s*)?[Aa](?:\s*\)|\s*[\.\-\:])\s*\S""",
        full_text, re.VERBOSE,
    )
    if opt_start:
        stem_text = full_text[:opt_start.start()].strip()
        opts_text = full_text[opt_start.start():]
    else:
        stem_text = full_text
        opts_text = ""

    stem_text = re.sub(r"\s+", " ", stem_text).strip()
    options   = _parse_options(opts_text) if opts_text else {}

    # Explanation
    explanation = ""
    exp_m = re.search(
        r"(?i)(?:explanation|rationale|discuss(?:ion)?)\s*[:\-]?\s*",
        full_text,
    )
    if exp_m:
        explanation = full_text[exp_m.end():].strip()
        explanation = re.sub(RE_ANS, "", explanation).strip()

    # Topic tags
    topic_tags = _extract_topic_tags(stem_text)

    # Flags
    flags: List[str] = []
    if not ans:
        flags.append("missing_answer")
    if not options.get("A"):
        flags.append("missing_options")
    if len(stem_text) < 10:
        flags.append("short_stem")

    return MCQ(
        question_id      = q_id,
        page             = page_idx + 1,
        question         = stem_text,
        question_html    = _to_html(stem_text),
        options          = options,
        answer           = ans or "",
        explanation      = explanation,
        explanation_html = _to_html(explanation),
        images           = [i for i in images if i],
        topic_tags       = topic_tags,
        schema_topics    = list(schema_topics),
        flags            = flags,
    )


# ══════════════════════════════════════════════════════════════════
# TOPIC TAG EXTRACTION
# ══════════════════════════════════════════════════════════════════

TOPIC_KWS: Dict[str, List[str]] = {
    "bacteria"        : ["bacteria","staph","strep","e.coli","klebsiell",
                         "pseudomonas","salmonell","shigell","clostridium",
                         "mycobacter","tuberculosis","bacill"],
    "virus"           : ["virus","viral","hiv","hepatitis","influenza",
                         "herpes","dengue","rabies","polio","measles",
                         "varicella","ebola","covid","sars","retrovir"],
    "fungi"           : ["fungi","fungal","candida","aspergill","crypto",
                         "histoplasm","dermatophyte","yeast","mould"],
    "parasite"        : ["parasite","protozoa","malaria","leishmania",
                         "trypanosoma","amoeba","giardia","helminth",
                         "worm","nematode","cestode","trematode"],
    "immunology"      : ["immuno","antibody","antigen","vaccine","immunity",
                         "complement","cytokine","lymphocyte","t cell",
                         "b cell","mhc","hla","interferon","interleukin"],
    "sterilization"   : ["steriliz","disinfect","autoclave","pasteuriz",
                         "antiseptic","decontam","fumigat"],
    "genetics"        : ["gene","genetic","dna","rna","chromosom","mutation",
                         "heredit","allele","pcr","plasmid","transposon"],
    "pharmacokinetics": ["pharmacokinet","absorption","distribution",
                         "metabolism","excretion","half.life","bioavailab"],
    "nerve"           : ["nerve","neural","neuron","plexus","ganglion"],
    "artery"          : ["artery","arterial","aorta","carotid","femoral"],
    "bone"            : ["bone","skeletal","vertebra","skull","pelvis",
                         "femur","tibia","fracture"],
}


def _extract_topic_tags(text: str) -> List[str]:
    t     = text.lower()
    found = []
    for tag, kws in TOPIC_KWS.items():
        if any(kw in t for kw in kws):
            found.append(tag)
    return found[:5]


# ══════════════════════════════════════════════════════════════════
# DICT CONVERTER
# ══════════════════════════════════════════════════════════════════

def _q_to_dict(mcq: MCQ) -> dict:
    """Convert MCQ dataclass to plain dict for JSON serialisation."""
    return {
        "question_id"      : mcq.question_id,
        "pdf_stem"         : mcq.pdf_stem,
        "subject"          : mcq.subject,
        "topic"            : mcq.topic,
        "pearl_id"         : mcq.pearl_id,
        "page"             : mcq.page,
        "question"         : mcq.question,
        "question_html"    : mcq.question_html,
        "question_image"   : mcq.question_image,
        "options"          : mcq.options,
        "answer"           : mcq.answer,
        "explanation"      : mcq.explanation,
        "explanation_html" : mcq.explanation_html,
        "explanation_image": mcq.explanation_image,
        "images"           : mcq.images,
        "topic_tags"       : mcq.topic_tags,
        "schema_topics"    : mcq.schema_topics,
        "flags"            : mcq.flags,
    }


# ══════════════════════════════════════════════════════════════════
# BLEEDING DETECTOR
# ══════════════════════════════════════════════════════════════════

def detect_bleeding(question: dict) -> List[str]:
    issues: List[str] = []
    stem = question.get("question", "")
    opts = question.get("options", {})
    exp  = question.get("explanation", "")

    if re.search(r"(?m)^\s*[A-D]\s*[\.\):]", stem):
        issues.append("bleed_stem_has_options")

    for letter, text in opts.items():
        others = [l for l in "ABCD" if l != letter]
        for other in others:
            if re.search(rf"(?m)^\s*{other}\s*[\.\):]", text or ""):
                issues.append(f"bleed_opt_{letter}_has_{other}")
                break

    if exp and re.match(r"^\s*[A-D]\s*[\.\):]", exp):
        issues.append("bleed_exp_starts_with_option")

    for letter, text in opts.items():
        if RE_ANS.search(text or ""):
            issues.append(f"bleed_opt_{letter}_has_answer")

    return issues


# ══════════════════════════════════════════════════════════════════
# MANIFEST BUILDER
# ══════════════════════════════════════════════════════════════════

def build_manifest(
    pdf_stem   : str,
    questions  : List[dict],
    source_path: str = "",
    dpi        : int = 300,
) -> dict:
    from datetime import datetime, timezone
    total   = len(questions)
    missing = sum(1 for q in questions if not q.get("answer"))
    bleed   = sum(
        1 for q in questions
        if any(f.startswith("bleed")
               for f in q.get("flags", []))
    )
    subjects = list({q.get("subject", "") for q in questions})

    return {
        "pdf_stem"    : pdf_stem,
        "source_path" : source_path,
        "dpi"         : dpi,
        "total_q"     : total,
        "missing_ans" : missing,
        "bleed_count" : bleed,
        "subjects"    : subjects,
        "status"      : "raw",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "edit_log"    : [],
    }
    