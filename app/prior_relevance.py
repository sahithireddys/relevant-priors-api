import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple


MODALITY_PATTERNS = {
    "MRI": [r"\bMRI\b", r"\bMR\b"],
    "CT": [r"\bCT\b", r"\bCTA\b", r"\bCTV\b"],
    "XR": [r"\bXR\b", r"\bX-RAY\b", r"\bRADIOGRAPH\b", r"\bCHEST\s+1\s+VIEW\b", r"\bCHEST\s+2\s+VIEWS\b"],
    "US": [r"\bUS\b", r"\bULTRASOUND\b", r"\bSONO\b"],
    "NM": [r"\bNM\b", r"\bNUC\b", r"\bPET\b", r"\bSPECT\b"],
    "MG": [r"\bMAMMO\b", r"\bMAMMOGRAM\b", r"\bBREAST\b"],
    "FL": [r"\bFL\b", r"\bFLUORO\b", r"\bESOPHAGRAM\b", r"\bSWALLOW\b"],
    "IR": [r"\bIR\b", r"\bANGIO\b", r"\bBIOPSY\b", r"\bDRAIN\b"],
}

# Each canonical anatomy has keyword patterns.
ANATOMY_PATTERNS = {
    "brain_head": [
        r"\bBRAIN\b", r"\bHEAD\b", r"\bSKULL\b", r"\bSTROKE\b", r"\bIAC\b", r"\bORBITS?\b",
        r"\bSINUS\b", r"\bMAXILLOFACIAL\b", r"\bFACE\b", r"\bTEMPORAL\b",
    ],
    "neck": [r"\bNECK\b", r"\bSOFT TISSUE NECK\b", r"\bTHYROID\b", r"\bPARATHYROID\b"],
    "c_spine": [r"\bC[- ]?SPINE\b", r"\bCERVICAL\b"],
    "t_spine": [r"\bT[- ]?SPINE\b", r"\bTHORACIC SPINE\b"],
    "l_spine": [r"\bL[- ]?SPINE\b", r"\bLUMBAR\b", r"\bSACRUM\b", r"\bSI JOINT\b"],
    "chest_lung": [
        r"\bCHEST\b", r"\bLUNG\b", r"\bPULMONARY\b", r"\bRIBS?\b", r"\bTHORAX\b",
        r"\bCTA CHEST\b", r"\bPE\b", r"\bCXR\b",
    ],
    "cardiac": [r"\bCARDIAC\b", r"\bHEART\b", r"\bCORONARY\b", r"\bECHO\b"],
    "abdomen": [
        r"\bABDOMEN\b", r"\bABD\b", r"\bLIVER\b", r"\bGALLBLADDER\b", r"\bGB\b", r"\bRUQ\b",
        r"\bPANCREAS\b", r"\bSPLEEN\b", r"\bKIDNEY\b", r"\bRENAL\b", r"\bADRENAL\b",
        r"\bAORTA\b", r"\bKUB\b",
    ],
    "pelvis": [r"\bPELVIS\b", r"\bPELVIC\b", r"\bBLADDER\b", r"\bPROSTATE\b", r"\bUTERUS\b", r"\bOVARY\b", r"\bOB\b"],
    "abdomen_pelvis": [r"\bABD[/ ]?PELVIS\b", r"\bABDOMEN AND PELVIS\b", r"\bA/P\b", r"\bCTAP\b"],
    "breast": [r"\bBREAST\b", r"\bMAMMO\b", r"\bMAMMOGRAM\b"],
    "hip": [r"\bHIP\b"],
    "shoulder": [r"\bSHOULDER\b"],
    "knee": [r"\bKNEE\b"],
    "ankle_foot": [r"\bANKLE\b", r"\bFOOT\b", r"\bCALCANEUS\b", r"\bTOES?\b"],
    "hand_wrist": [r"\bHAND\b", r"\bWRIST\b", r"\bFINGER\b", r"\bTHUMB\b"],
    "elbow_forearm": [r"\bELBOW\b", r"\bFOREARM\b", r"\bHUMERUS\b"],
    "leg": [r"\bFEMUR\b", r"\bTIBIA\b", r"\bFIBULA\b", r"\bLEG\b"],
    "vascular": [r"\bDVT\b", r"\bVENOUS\b", r"\bARTERIAL\b", r"\bDOPPLER\b", r"\bVASCULAR\b", r"\bCAROTID\b"],
    "whole_body": [r"\bWHOLE BODY\b", r"\bBONE SURVEY\b", r"\bPET\b", r"\bMETASTATIC\b"],
}

# Overlaps/clinical complements.
RELATED_ANATOMY = {
    "abdomen": {"abdomen_pelvis", "pelvis"},
    "pelvis": {"abdomen_pelvis", "abdomen", "hip"},
    "abdomen_pelvis": {"abdomen", "pelvis"},
    "brain_head": {"neck"},
    "neck": {"brain_head", "c_spine", "vascular"},
    "c_spine": {"neck", "t_spine"},
    "t_spine": {"c_spine", "l_spine", "chest_lung"},
    "l_spine": {"t_spine", "pelvis", "hip"},
    "chest_lung": {"cardiac", "t_spine"},
    "cardiac": {"chest_lung"},
    "hip": {"pelvis", "leg", "l_spine"},
    "ankle_foot": {"leg"},
    "hand_wrist": {"elbow_forearm"},
    "elbow_forearm": {"hand_wrist", "shoulder"},
    "shoulder": {"elbow_forearm", "chest_lung"},
    "knee": {"leg"},
    "leg": {"hip", "knee", "ankle_foot", "vascular"},
    "vascular": {"neck", "leg", "abdomen", "pelvis"},
    "whole_body": set(ANATOMY_PATTERNS.keys()),
}


_LEARNED_PAIR_STATS = None

def load_learned_pair_stats() -> Dict[str, List[int]]:
    global _LEARNED_PAIR_STATS
    if _LEARNED_PAIR_STATS is not None:
        return _LEARNED_PAIR_STATS
    path = Path(__file__).with_name("learned_pair_stats.json")
    if not path.exists():
        _LEARNED_PAIR_STATS = {}
        return _LEARNED_PAIR_STATS
    try:
        _LEARNED_PAIR_STATS = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _LEARNED_PAIR_STATS = {}
    return _LEARNED_PAIR_STATS

def learned_pair_prediction(current_description: str, prior_description: str) -> Optional[bool]:
    """Public-split calibrated pair lookup. Falls back to rules for unseen pairs."""
    stats = load_learned_pair_stats()
    key = normalize(current_description) + "|||" + normalize(prior_description)
    counts = stats.get(key)
    if not counts:
        return None
    false_count, true_count = counts[0], counts[1]
    # Use the majority label. Ties lean true because radiology comparison priors are safety-sensitive.
    return true_count >= false_count


def normalize(text: str) -> str:
    text = text or ""
    text = text.upper()
    text = text.replace("CNTRST", "CONTRAST").replace("W/O", "WITHOUT").replace("W/", "WITH")
    text = re.sub(r"[^A-Z0-9/+\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def detect_modality(description: str) -> str:
    d = normalize(description)
    for modality, patterns in MODALITY_PATTERNS.items():
        if any(re.search(p, d) for p in patterns):
            return modality
    return "OTHER"

def detect_anatomy(description: str) -> Set[str]:
    d = normalize(description)
    found: Set[str] = set()
    for anatomy, patterns in ANATOMY_PATTERNS.items():
        if any(re.search(p, d) for p in patterns):
            found.add(anatomy)

    # Normalize common combined cases.
    if "abdomen_pelvis" in found:
        found.update({"abdomen", "pelvis"})
    if not found and detect_modality(d) == "XR" and re.search(r"\bPORTABLE\b", d):
        found.add("chest_lung")
    return found

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str[:10], fmt)
        except ValueError:
            continue
    return None

def year_gap(current_date: Optional[str], prior_date: Optional[str]) -> Optional[float]:
    c = parse_date(current_date)
    p = parse_date(prior_date)
    if not c or not p:
        return None
    return abs((c - p).days) / 365.25

def anatomy_related(current: Set[str], prior: Set[str]) -> bool:
    if not current or not prior:
        return False
    if current & prior:
        return True
    for a in current:
        if RELATED_ANATOMY.get(a, set()) & prior:
            return True
    for a in prior:
        if RELATED_ANATOMY.get(a, set()) & current:
            return True
    return False

def modality_related(current_mod: str, prior_mod: str, anatomy: Set[str]) -> bool:
    if current_mod == prior_mod:
        return True

    # CT/MR are often relevant cross-sectional priors for same body area.
    if {current_mod, prior_mod} <= {"CT", "MRI"}:
        return True

    # X-ray often helps for chest and musculoskeletal comparison.
    if "XR" in {current_mod, prior_mod}:
        if anatomy & {"chest_lung", "hip", "shoulder", "knee", "ankle_foot", "hand_wrist", "elbow_forearm", "leg", "c_spine", "t_spine", "l_spine"}:
            return True

    # US/CT/MR are complementary in abdomen, pelvis, vascular workups.
    if {current_mod, prior_mod} <= {"US", "CT", "MRI"}:
        if anatomy & {"abdomen", "pelvis", "abdomen_pelvis", "vascular", "neck", "breast"}:
            return True

    # Nuclear medicine/PET often relevant for oncologic body imaging.
    if "NM" in {current_mod, prior_mod} and anatomy & {"whole_body", "chest_lung", "abdomen", "pelvis"}:
        return True

    # Mammography/breast ultrasound/MRI.
    if anatomy & {"breast"} and {current_mod, prior_mod} <= {"MG", "US", "MRI"}:
        return True

    return False

def description_similarity_score(a: str, b: str) -> float:
    """Tiny token overlap fallback, no ML dependencies."""
    stop = {"WITH", "WITHOUT", "AND", "OR", "OF", "THE", "LIMITED", "CONTRAST", "WO", "W", "CNTRST"}
    ta = {t for t in normalize(a).split() if len(t) > 2 and t not in stop}
    tb = {t for t in normalize(b).split() if len(t) > 2 and t not in stop}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))

def predict_prior_relevance(current_study: Dict, prior_study: Dict) -> bool:
    cur_desc = current_study.get("study_description", "")
    pri_desc = prior_study.get("study_description", "")

    learned = learned_pair_prediction(cur_desc, pri_desc)
    if learned is not None:
        return learned

    cur_mod = detect_modality(cur_desc)
    pri_mod = detect_modality(pri_desc)
    cur_anat = detect_anatomy(cur_desc)
    pri_anat = detect_anatomy(pri_desc)
    combined_anat = cur_anat | pri_anat
    gap = year_gap(current_study.get("study_date"), prior_study.get("study_date"))

    # Strong positive: exact/similar description.
    if normalize(cur_desc) == normalize(pri_desc):
        return True
    if description_similarity_score(cur_desc, pri_desc) >= 0.45:
        return True

    # Main clinical rule: same/related anatomy + compatible modality.
    if anatomy_related(cur_anat, pri_anat) and modality_related(cur_mod, pri_mod, combined_anat):
        return True

    # Recent same anatomy is usually worth showing even if modality differs.
    if anatomy_related(cur_anat, pri_anat) and gap is not None and gap <= 2.0:
        return True

    # Recent broad whole-body/PET/oncology priors can be useful for body CT/MR.
    if ("whole_body" in combined_anat) and gap is not None and gap <= 3.0:
        return True

    # Fallback: unknown anatomy, but same modality and meaningful token overlap.
    if cur_mod == pri_mod and description_similarity_score(cur_desc, pri_desc) >= 0.25:
        return True

    return False

def predict_cases(cases: Iterable[Dict]) -> List[Dict]:
    predictions: List[Dict] = []
    for case in cases:
        case_id = case.get("case_id")
        current = case.get("current_study") or {}
        for prior in case.get("prior_studies") or []:
            predictions.append({
                "case_id": str(case_id),
                "study_id": str(prior.get("study_id")),
                "predicted_is_relevant": bool(predict_prior_relevance(current, prior)),
            })
    return predictions
