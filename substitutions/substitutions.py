"""
Substitution Planner for Book Questions → Candidate Questions

This module loads three CSVs, computes per-container target levels, and builds a
substitution plan for book questions using unassigned candidate questions.

Key behaviors:
- If a book question's subject is "הסקה מתרשים" → always add a row with empty new_question_id.
- If a book question's *container* appears in containers_to_be_removed.csv → skip substitution.
- Otherwise: prefer a same-subject candidate whose level is closest to the target level:
    target level = container average (if available) → original question level → global median.
- Each candidate question can be used at most once.
- Exports results with UTF-8 BOM (utf-8-sig) for Excel-friendly Hebrew.

Design goals:
- Readability first, then runtime. No code duplication. Small, well-named helpers.
- Robust ID canonicalization (e.g., "123.0" → "123") applied consistently.
- Gentle fallback behavior when data is missing or messy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, List

import numpy as np
import pandas as pd


# =======================
# ====== CONSTANTS  =====
# =======================

# Default file names (will also be probed in /mnt/data if not found in CWD)
FILE_BOOK = "book_qusetions.csv"             # note: original misspelling preserved intentionally
FILE_CONTAINERS_QUESTIONS = "containers_questions.csv"
FILE_ASSIGNMENTS_CONTAINERS = "assignments_containers.csv"
FILE_CONTAINERS_TO_BE_REMOVED = "containers_to_be_removed.csv"

OUTPUT_CSV = "substitution_plan.csv"

# Column name schema — adjust here if your CSV headers change
@dataclass(frozen=True)
class Schema:
    # book_questions
    BOOK_QID: str = "q_id"
    BOOK_MISSING_FLAG: str = "missing in book"

    # containers_questions
    CQ_QID: str = "Question ID"
    CQ_CONTAINER: str = "Container ID"
    CQ_LEVEL: str = "Question Level"
    CQ_SUBJECT: str = "Question (Main) Tag"
    CQ_ACTIVE: str = "Question Active"


SCHEMA = Schema()

# Special-case subject (normalized, i.e., lowercased & stripped) that should never be substituted
SPECIAL_SUBJECT_SKIP_NORM = "הסקה מתרשים"

# CSV read defaults
CSV_READ_KW = dict(encoding="utf-8", low_memory=False)

# CSV write encoding for Excel-friendly Hebrew
CSV_WRITE_ENCODING = "utf-8-sig"


# =======================
# ====== UTILITIES  =====
# =======================

def resolve_path(filename: str) -> Path:
    """
    Return a Path to `filename`, preferring CWD; fallback to /mnt/data for notebook/tooling contexts.
    """
    p = Path(filename)
    if p.exists():
        return p
    alt = Path("/mnt/data") / filename
    return alt if alt.exists() else p


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Wrapper around pandas.read_csv with sane defaults and a consistent place to customize."""
    kw = dict(CSV_READ_KW)
    kw.update(kwargs)
    return pd.read_csv(path, **kw)


def to_int_like_str_series(s: pd.Series) -> pd.Series:
    """
    Canonicalize IDs:
    - Strip whitespace, keep as string
    - If numeric & integral (e.g., "123.0"), convert to "123" (string)
    """
    s = s.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    intlike = numeric.notna() & ((numeric % 1) == 0)
    s.loc[intlike] = numeric.loc[intlike].astype("Int64").astype(str)
    return s


def normalize_subject_series(s: pd.Series) -> pd.Series:
    """
    Normalize Hebrew/English subject column:
    - Cast to string
    - Strip
    - Lowercase
    """
    return s.astype(str).str.strip().str.lower()


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Convert to numeric with NaN for invalid values."""
    return pd.to_numeric(s, errors="coerce")


def has_container_mask(container_col: pd.Series) -> pd.Series:
    """
    Rows that *do* have a valid container:
    - Not NA
    - Not empty
    - Numeric value != 0 (we permit strings, so we coerce)
    """
    return (
        container_col.notna()
        & container_col.astype(str).str.strip().ne("")
        & (pd.to_numeric(container_col, errors="coerce") != 0)
    )


def no_container_mask(container_col: pd.Series) -> pd.Series:
    """Inverse of has_container_mask."""
    return ~has_container_mask(container_col)


def is_active_mask(active_col: pd.Series) -> pd.Series:
    """
    Interpret 'active' flexibly:
    - numeric 1 → active
    - strings in {"1","true","t","yes","y"} → active
    """
    numeric_one = (pd.to_numeric(active_col, errors="coerce") == 1)
    truthy = active_col.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    return numeric_one | truthy


def median_or_nan(values: pd.Series) -> float:
    """Return float median or NaN if empty."""
    return float(values.median()) if len(values) > 0 else np.nan

def pick_best_candidate_idx(
    levels: np.ndarray,
    available_mask: np.ndarray,
    target: float,
) -> Optional[int]:
    """
    Return the index of the available candidate whose level is closest to `target`.
    If no candidates are available, return None.

    Notes:
    - Fully vectorized (no Python loops).
    - If `target` is NaN, we select the *first* available candidate (no preference).
    """
    if not np.any(available_mask):
        return None

    # If target is NaN, just pick the first available candidate deterministically.
    if target is None or (isinstance(target, float) and np.isnan(target)):
        return int(np.flatnonzero(available_mask)[0])

    # Compute absolute diffs only for available items; others set to +inf
    diffs = np.full_like(levels, np.inf, dtype=float)
    diffs[available_mask] = np.abs(levels[available_mask] - target)

    return int(np.argmin(diffs))


def _format_signed_delta(x: float | None) -> str:
    """Format a signed delta with 2 decimals; empty if None/NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:+.2f}"


def _normalize_container_membership_set(raw_ids: Iterable[object]) -> tuple[set[str], set[int]]:
    """
    Return two sets for fast membership:
    - string set (trimmed, BOM-stripped), including both 'N' and 'N.0' variants for int-like ids
    - int set (for int-like IDs)
    """
    str_set: set[str] = set()
    int_set: set[int] = set()
    for v in raw_ids:
        if pd.isna(v):
            continue
        s = str(v).replace("\ufeff", "").strip()
        if not s:
            continue

        # Always include the raw cleaned string
        str_set.add(s)

        # If numeric/int-like, include extra normalized forms
        try:
            i = int(float(s))
            int_set.add(i)
            str_set.add(str(i))             # '62616'
            str_set.add(f"{float(i):.1f}")  # '62616.0'
        except Exception:
            pass

        # If the value ends with '.0', also include the trimmed form
        if s.endswith(".0"):
            str_set.add(s[:-2])
    return str_set, int_set



def _series_is_in_containers(series: pd.Series, *, str_set: set[str], int_set: set[int]) -> pd.Series:
    """Efficient membership test for container IDs against both string and int-like forms."""
    s_str = series.astype(str).str.strip()
    s_num = pd.to_numeric(series, errors="coerce")
    mask_str = s_str.isin(str_set)
    mask_int = pd.Series(False, index=series.index)
    if len(int_set) > 0:
        s_int = s_num.astype("Int64")
        mask_int = s_int.isin(list(int_set))
    return mask_str | mask_int


# ==============================
# ====== CORE CALCULATIONS =====
# ==============================

@dataclass
class Datasets:
    book_questions: pd.DataFrame
    containers_questions: pd.DataFrame
    assignments_containers: pd.DataFrame


def load_datasets(
    book_csv: Path,
    containers_questions_csv: Path,
    assignments_containers_csv: Path,
    **csv_kwargs,
) -> Datasets:
    """
    Load the three CSV files into separate DataFrames.

    Any extra pandas.read_csv kwargs can be passed via `csv_kwargs`.
    """
    kw = dict(CSV_READ_KW)
    kw.update(csv_kwargs)
    return Datasets(
        book_questions=read_csv(book_csv, **kw),
        containers_questions=read_csv(containers_questions_csv, **kw),
        assignments_containers=read_csv(assignments_containers_csv, **kw),
    )


def average_question_level_by_container(
    containers_questions: pd.DataFrame,
    *,
    container_col: str = SCHEMA.CQ_CONTAINER,
    level_col: str = SCHEMA.CQ_LEVEL,
    only_active: bool = False,
    active_col: str = SCHEMA.CQ_ACTIVE,
    round_to: int | None = 2,
) -> Dict[int, float]:
    """
    Compute average (mean) question level per container.
    """
    df = containers_questions.copy()

    # active filter
    if only_active and (active_col in df.columns):
        df = df[is_active_mask(df[active_col])]

    # numeric levels only
    df[level_col] = coerce_numeric_series(df[level_col])
    df = df.dropna(subset=[level_col, container_col])

    means = df.groupby(container_col, dropna=False)[level_col].mean()
    if round_to is not None:
        means = means.round(round_to)

    # Try to int-cast index for stable keys (e.g., "12" → 12)
    try:
        means.index = means.index.astype(float).astype(int)
    except Exception:
        pass

    # Best-effort coercion to float for values:
    means = means.astype(float, errors="ignore")
    return means.to_dict()


def count_questions_per_container(
    containers_questions: pd.DataFrame,
    *,
    container_col: str = SCHEMA.CQ_CONTAINER,
    level_col: str = SCHEMA.CQ_LEVEL,
    only_active: bool = False,
    active_col: str = SCHEMA.CQ_ACTIVE,
) -> Dict[int, int]:
    """
    Count how many items contribute to each container's average.
    Filters match average_question_level_by_container.
    """
    df = containers_questions.copy()
    if only_active and (active_col in df.columns):
        df = df[is_active_mask(df[active_col])]

    df[level_col] = coerce_numeric_series(df[level_col])
    df = df.dropna(subset=[level_col, container_col])

    counts = df.groupby(container_col, dropna=False)[level_col].size()

    try:
        counts.index = counts.index.astype(float).astype(int)
    except Exception:
        pass

    return counts.to_dict()


def map_book_qids(  # helper used by multiple steps
    book_df: pd.DataFrame,
    *,
    book_question_id_col: str = SCHEMA.BOOK_QID,
    book_missing_col: str = SCHEMA.BOOK_MISSING_FLAG,
) -> pd.Series:
    """
    Return a canonicalized Series of eligible book question IDs (as strings),
    excluding rows where 'missing in book' == 1, and excluding empty/NaN IDs.
    """
    df = book_df.copy()
    mask_valid = df[book_question_id_col].notna() & df[book_question_id_col].astype(str).str.strip().ne("")
    if book_missing_col in df.columns:
        mask_valid &= (coerce_numeric_series(df[book_missing_col]) != 1)

    s = df.loc[mask_valid, book_question_id_col]
    return to_int_like_str_series(s)


def build_assigned_lookup(
    cq_df: pd.DataFrame,
    *,
    cq_question_id_col: str = SCHEMA.CQ_QID,
    cq_container_col: str = SCHEMA.CQ_CONTAINER,
    subject_col: str = SCHEMA.CQ_SUBJECT,
    level_col: str = SCHEMA.CQ_LEVEL,
) -> pd.DataFrame:
    """
    Build a lookup DataFrame indexed by canonicalized question ID for rows that *are assigned* to a container.

    The resulting frame has columns: [cq_container_col, subject_col, level_col].
    """
    assigned = cq_df.copy()
    mask = has_container_mask(assigned[cq_container_col]) & assigned[cq_question_id_col].notna()
    assigned = assigned.loc[mask, [cq_question_id_col, cq_container_col, subject_col, level_col]].copy()

    # Canonicalize IDs and normalize types
    assigned[cq_question_id_col] = to_int_like_str_series(assigned[cq_question_id_col])
    assigned[cq_container_col] = to_int_like_str_series(assigned[cq_container_col])
    assigned[level_col] = coerce_numeric_series(assigned[level_col])
    if subject_col in assigned.columns:
        assigned[subject_col] = assigned[subject_col].astype(str).str.strip()

    # Drop duplicates by question id; keep first
    assigned = assigned.drop_duplicates(subset=[cq_question_id_col], keep="first")
    return assigned.set_index(cq_question_id_col)


def load_removed_container_ids(
    path: Optional[Path] = None,
    *,
    container_col: str = SCHEMA.CQ_CONTAINER,
) -> set:
    """
    Load container IDs to be removed.
    Robust to: missing file, empty file, headerless files, header-only files, and BOM on first value.
    Returns a set of strings.
    """
    # Resolve path candidates
    candidates = []
    if path is not None:
        candidates.append(path)
    else:
        candidates.append(resolve_path(FILE_CONTAINERS_TO_BE_REMOVED))

    csv_path = next((p for p in candidates if p is not None and p.exists()), None)
    if csv_path is None:
        return set()

    # First try: normal read
    try:
        df = read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return set()

    values: List[str] = []

    def _clean_series(ser: pd.Series) -> List[str]:
        return (
            ser.dropna()
              .astype(str)
              .str.replace("\ufeff", "", regex=False)  # strip BOM if present
              .str.strip()
              .tolist()
        )

    # Case A: we have rows
    if not df.empty:
        if container_col in df.columns and not df[container_col].empty:
            values = _clean_series(df[container_col])
        else:
            # Fall back to first column
            values = _clean_series(df.iloc[:, 0])
    else:
        # Case B: header-only CSV → use column names as values
        header_values = [str(c).replace("\ufeff", "").strip() for c in df.columns if str(c).strip()]
        values = header_values

        # As an extra fallback, try header=None
        if not values:
            try:
                df2 = read_csv(csv_path, header=None, names=[container_col])
                if not df2.empty:
                    values = _clean_series(df2.iloc[:, 0])
            except pd.errors.EmptyDataError:
                values = []

    return set(v for v in values if v != "")


def build_candidate_pool(
    cq_df: pd.DataFrame,
    *,
    book_qids: Iterable[str],
    only_active_candidates: bool = True,
    cq_question_id_col: str = SCHEMA.CQ_QID,
    cq_container_col: str = SCHEMA.CQ_CONTAINER,
    level_col: str = SCHEMA.CQ_LEVEL,
    subject_col: str = SCHEMA.CQ_SUBJECT,
    active_col: str = SCHEMA.CQ_ACTIVE,
    removed_container_ids: Optional[Iterable[object]] = None,   # NEW
) -> Tuple[pd.DataFrame, float]:
    """
    Return a candidate pool DataFrame of *unassigned* questions (no container) OR
    questions assigned to a container slated for removal. Only include valid rows
    (have QID, subject, level), and exclude any QID that appears in the book list.

    The returned DataFrame has:
        - cq_question_id_col (string, canonicalized)
        - subject_col (normalized lowercase)
        - level_col (float)
        - 'used' boolean column initialized to False

    Also returns the global median of candidate levels (float; NaN if empty).
    """
    df = cq_df.copy()

    # Active filter if requested
    if only_active_candidates and (active_col in df.columns):
        df = df[is_active_mask(df[active_col])]

    # Keep unassigned OR assigned-to-a-removed-container
    mask_unassigned = no_container_mask(df[cq_container_col])

    if removed_container_ids:
        str_set, int_set = _normalize_container_membership_set(removed_container_ids)
        mask_removed_container = has_container_mask(df[cq_container_col]) & _series_is_in_containers(
            df[cq_container_col], str_set=str_set, int_set=int_set
        )
    else:
        mask_removed_container = pd.Series(False, index=df.index)

    df = df[mask_unassigned | mask_removed_container]

    # Basic validity
    mask_valid = (
        df[cq_question_id_col].notna()
        & df[subject_col].notna()
        & df[level_col].notna()
    )
    df = df.loc[mask_valid, [cq_question_id_col, subject_col, level_col]].copy()

    # Canonicalize and normalize
    df[cq_question_id_col] = to_int_like_str_series(df[cq_question_id_col])
    df[subject_col] = normalize_subject_series(df[subject_col])
    df[level_col] = coerce_numeric_series(df[level_col])

    # Exclude book qids (don't reuse originals)
    df = df[~df[cq_question_id_col].isin(set(book_qids))].copy()

    # Initialize usage tracking
    df["used"] = False

    global_median = median_or_nan(df[level_col]) if not df.empty else np.nan
    return df, global_median


def get_target_level(
    *,
    container_id: object,
    original_level: Optional[float],
    avg_by_container: Optional[Dict[object, float]],
    global_median: float,
) -> float:
    """
    Determine the target level to minimize distance to:
    1) avg_by_container[container_id] if available
    2) original_level
    3) global_median
    """
    # 1) Container average
    if avg_by_container:
        # tolerate str vs int mismatches for keys
        if container_id in avg_by_container:
            return float(avg_by_container[container_id])
        # try int-cast variant (e.g., "12" vs 12)
        try:
            key2 = int(float(container_id))
            if key2 in avg_by_container:
                return float(avg_by_container[key2])
        except Exception:
            pass

    # 2) original level
    if original_level is not None and not np.isnan(original_level):
        return float(original_level)

    # 3) global median (may still be NaN if candidate pool empty)
    return float(global_median)


# 1) Eligible book IDs

def _eligible_book_qids(
    book_df: pd.DataFrame,
    *,
    book_question_id_col: str = SCHEMA.BOOK_QID,
    book_missing_col: str = SCHEMA.BOOK_MISSING_FLAG,
) -> List[str]:
    """Step 1: Determine eligible BOOK question IDs (non-empty, not 'missing in book')."""
    s = map_book_qids(
        book_df,
        book_question_id_col=book_question_id_col,
        book_missing_col=book_missing_col,
    )
    return s.unique().tolist()


# 2) Assigned lookup

def _assigned_lookup(
    cq_df: pd.DataFrame,
    *,
    cq_question_id_col: str = SCHEMA.CQ_QID,
    cq_container_col: str = SCHEMA.CQ_CONTAINER,
    subject_col: str = SCHEMA.CQ_SUBJECT,
    level_col: str = SCHEMA.CQ_LEVEL,
) -> pd.DataFrame:
    """Step 2: Build ASSIGNED lookup (q_id → (container, subject_raw, level))."""
    return build_assigned_lookup(
        cq_df,
        cq_question_id_col=cq_question_id_col,
        cq_container_col=cq_container_col,
        subject_col=subject_col,
        level_col=level_col,
    )


# 3) Candidate pool & fast-access structs

def _prepare_candidate_structs(
    cq_df: pd.DataFrame,
    *,
    book_qids: List[str],
    only_active_candidates: bool,
    cq_question_id_col: str = SCHEMA.CQ_QID,
    cq_container_col: str = SCHEMA.CQ_CONTAINER,
    level_col: str = SCHEMA.CQ_LEVEL,
    subject_col: str = SCHEMA.CQ_SUBJECT,
    active_col: str = SCHEMA.CQ_ACTIVE,
    removed_container_ids: Optional[Iterable[object]] = None,   # NEW
) -> Tuple[pd.DataFrame, dict, Dict[str, np.ndarray], float]:
    """
    Step 3: Build the CANDIDATE pool and derive fast vectors/masks for quick selection.
    """
    candidates, global_median = build_candidate_pool(
        cq_df,
        book_qids=book_qids,
        only_active_candidates=only_active_candidates,
        cq_question_id_col=cq_question_id_col,
        cq_container_col=cq_container_col,
        level_col=level_col,
        subject_col=subject_col,
        active_col=active_col,
        removed_container_ids=removed_container_ids,  # pass-through
    )

    qids = candidates[cq_question_id_col].to_numpy()
    levels = candidates[level_col].to_numpy(dtype=float)
    subjects = candidates[subject_col].to_numpy()
    used = candidates["used"].to_numpy()  # boolean view; mutate to mark usage

    # Precompute subject masks
    subject_to_mask: Dict[str, np.ndarray] = {}
    if len(candidates) > 0:
        for subj in np.unique(subjects):
            subject_to_mask[subj] = (subjects == subj)

    arrays = {"qids": qids, "levels": levels, "subjects": subjects, "used": used}
    return candidates, arrays, subject_to_mask, float(global_median)


# 4) Substitution row generation (main loop)

def _generate_substitution_rows(
    book_qids: List[str],
    assigned: pd.DataFrame,
    arrays: dict,
    subject_to_mask: Dict[str, np.ndarray],
    *,
    cq_container_col: str = SCHEMA.CQ_CONTAINER,
    level_col: str = SCHEMA.CQ_LEVEL,
    subject_col: str = SCHEMA.CQ_SUBJECT,
    avg_by_container: Optional[Dict[object, float]],
    global_median: float,
    counts_by_container: Optional[Dict[object, int]] = None,   # for avg_delta
    removed_container_ids: Optional[Iterable[object]] = None,  # NEW
) -> List[dict]:
    """
    Step 4: For each eligible book qid present in ASSIGNED:
      - Skip substitution entirely if the container is slated for removal.
      - Respect the 'הסקה מתרשים' special case (no substitute).
      - Compute target level (container avg → original level → global median).
      - Pick best same-subject unused candidate; else best overall unused.
      - Mark chosen candidate as used and append a result row (dict).
    """
    cand_qids = arrays["qids"]
    cand_levels = arrays["levels"]
    cand_subjects = arrays["subjects"]
    cand_used = arrays["used"]

    # --- helpers --------------------------------------------------------------
    def _get_count(cid: object) -> int | None:
        if not counts_by_container:
            return None
        if cid in counts_by_container:
            return int(counts_by_container[cid])
        try:
            key2 = int(float(cid))
            return int(counts_by_container.get(key2)) if key2 in counts_by_container else None
        except Exception:
            return None

    # Normalize fast membership for removed containers
    removed_strs: set[str] = set()
    removed_ints: set[int] = set()
    if removed_container_ids:
        for v in removed_container_ids:
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            removed_strs.add(s)
            try:
                removed_ints.add(int(float(s)))
            except Exception:
                pass

    def _in_removed_containers(cid: object) -> bool:
        if cid is None or (isinstance(cid, float) and np.isnan(cid)):
            return False
        s = str(cid).replace("\ufeff", "").strip()
        if s in removed_strs:
            return True
        # Normalize 'N.0' → 'N' as an extra check
        if s.endswith(".0") and s[:-2] in removed_strs:
            return True
        try:
            i = int(float(s))
            if i in removed_ints:
                return True
        except Exception:
            pass
        return False

    results: List[dict] = []

    # --- main loop ------------------------------------------------------------
    for old_qid in book_qids:
        if old_qid not in assigned.index:
            continue

        row = assigned.loc[old_qid]
        container_id = row[cq_container_col]

        # 0) Skip entirely if this container is to be removed
        if _in_removed_containers(container_id):
            continue

        raw_subject = "" if pd.isna(row[subject_col]) else str(row[subject_col]).strip()
        norm_subject = raw_subject.lower()

        # 1) Special: never search for this subject
        if norm_subject == SPECIAL_SUBJECT_SKIP_NORM:
            results.append(
                {
                    "container_id": container_id,
                    "old_question_id": old_qid,
                    "new_question_id": "",
                    "no_substitute_in_subject": raw_subject,
                    "avg_delta": "",
                }
            )
            continue

        # 2) Levels
        try:
            orig_level = float(row[level_col])
        except Exception:
            orig_level = np.nan

        target = get_target_level(
            container_id=container_id,
            original_level=orig_level,
            avg_by_container=avg_by_container,
            global_median=global_median,
        )

        # 3) Try same-subject first
        pick_idx = None
        picked_from_same_subject = False
        same_mask = subject_to_mask.get(norm_subject)

        if same_mask is not None:
            available_same = (~cand_used) & same_mask
            pick_idx = pick_best_candidate_idx(cand_levels, available_same, target)
            picked_from_same_subject = pick_idx is not None

        # 4) Fallback to best overall unused
        if pick_idx is None:
            available_all = (~cand_used)
            pick_idx = pick_best_candidate_idx(cand_levels, available_all, target)

        if pick_idx is None:
            # No candidates available at all
            continue

        new_qid = cand_qids[pick_idx]
        new_level = float(cand_levels[pick_idx])
        cand_used[pick_idx] = True  # mark chosen candidate as used

        # 5) Δavg = (new_level - orig_level) / N
        N = _get_count(container_id)
        delta = (new_level - orig_level) / N if (N and not np.isnan(orig_level)) else None
        delta_str = _format_signed_delta(delta)

        results.append(
            {
                "container_id": container_id,
                "old_question_id": old_qid,
                "new_question_id": new_qid,
                "no_substitute_in_subject": ("" if picked_from_same_subject else raw_subject),
                "avg_delta": delta_str,
            }
        )

    return results


# 5) Assemble final DataFrame

def _rows_to_substitution_df(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["container_id", "old_question_id", "new_question_id", "no_substitute_in_subject", "avg_delta"],
    )
    # Make container_id display clean (e.g., '62616' not '62616.0')
    if "container_id" in df.columns:
        df["container_id"] = to_int_like_str_series(df["container_id"])
    return df


# Orchestrator (thin wrapper)

def build_substitution_df(
    book_df: pd.DataFrame,
    cq_df: pd.DataFrame,
    *,
    book_question_id_col: str = SCHEMA.BOOK_QID,
    book_missing_col: str = SCHEMA.BOOK_MISSING_FLAG,
    cq_question_id_col: str = SCHEMA.CQ_QID,
    cq_container_col: str = SCHEMA.CQ_CONTAINER,
    level_col: str = SCHEMA.CQ_LEVEL,
    subject_col: str = SCHEMA.CQ_SUBJECT,
    active_col: str = SCHEMA.CQ_ACTIVE,
    only_active_candidates: bool = True,
    avg_by_container: Optional[Dict[object, float]] = None,
    removed_container_ids: Optional[Iterable[object]] = None,  # NEW
) -> pd.DataFrame:
    """
    Build the substitution plan via 5 focused steps + removal logic.
    """
    # 1) Eligible book IDs
    book_qids = _eligible_book_qids(
        book_df,
        book_question_id_col=book_question_id_col,
        book_missing_col=book_missing_col,
    )
    if not book_qids:
        return _rows_to_substitution_df([])

    # 2) Assigned lookup
    assigned = _assigned_lookup(
        cq_df,
        cq_question_id_col=cq_question_id_col,
        cq_container_col=cq_container_col,
        subject_col=subject_col,
        level_col=level_col,
    )

    # 3) Candidate pool & structs
    _candidates, arrays, subject_to_mask, global_median = _prepare_candidate_structs(
        cq_df,
        book_qids=book_qids,
        only_active_candidates=only_active_candidates,
        cq_question_id_col=cq_question_id_col,
        cq_container_col=cq_container_col,
        level_col=level_col,
        subject_col=subject_col,
        active_col=active_col,
        removed_container_ids=removed_container_ids,  # pass-through
    )

    # NEW: counts for Δavg (match filters used for averages)
    counts_by_container = count_questions_per_container(
        cq_df,
        container_col=cq_container_col,
        level_col=level_col,
        only_active=True,
        active_col=active_col,
    )

    # 4) Generate rows
    rows = _generate_substitution_rows(
        book_qids,
        assigned,
        arrays,
        subject_to_mask,
        cq_container_col=cq_container_col,
        level_col=level_col,
        subject_col=subject_col,
        avg_by_container=avg_by_container,
        global_median=global_median,
        counts_by_container=counts_by_container,
        removed_container_ids=removed_container_ids,  # NEW
    )

    # 5) Assemble DataFrame
    subs_df = _rows_to_substitution_df(rows)

    # Extra safety: drop any rows for removed containers from the final output
    if removed_container_ids:
        str_set, int_set = _normalize_container_membership_set(removed_container_ids)
        keep_mask = ~_series_is_in_containers(subs_df["container_id"], str_set=str_set, int_set=int_set)
        subs_df = subs_df.loc[keep_mask].reset_index(drop=True)

    return subs_df


def count_book_questions_in_containers(
    book_df: pd.DataFrame,
    cq_df: pd.DataFrame,
    *,
    book_question_id_col: str = SCHEMA.BOOK_QID,
    book_missing_col: str = SCHEMA.BOOK_MISSING_FLAG,
    cq_question_id_col: str = SCHEMA.CQ_QID,
    cq_container_col: str = SCHEMA.CQ_CONTAINER,
) -> int:
    """
    Return the number of book question IDs that also appear as assigned (have a container) in containers_questions.
    """
    book_ids = set(map_book_qids(
        book_df,
        book_question_id_col=book_question_id_col,
        book_missing_col=book_missing_col,
    ).tolist())

    cq = cq_df.copy()
    mask = has_container_mask(cq[cq_container_col]) & cq[cq_question_id_col].notna()
    aq = cq.loc[mask, cq_question_id_col].copy()
    aq = to_int_like_str_series(aq)
    container_ids = set(aq.tolist())

    return len(book_ids & container_ids)


# =======================
# ====== PIPELINE   =====
# =======================

def run_pipeline(
    book_csv: Path,
    containers_questions_csv: Path,
    assignments_containers_csv: Path,
    out_csv: Path,
    *,
    only_active_candidates: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, float]]:
    """
    Execute the end-to-end pipeline:
    - Load datasets
    - Compute container averages (active-only)
    - Load containers-to-be-removed (optional)
    - Build substitution plan
    - Save CSV with utf-8-sig
    - Return (subs_df, avg_by_container) for further inspection
    """
    datasets = load_datasets(
        book_csv=book_csv,
        containers_questions_csv=containers_questions_csv,
        assignments_containers_csv=assignments_containers_csv,
    )

    avg_by_container = average_question_level_by_container(
        datasets.containers_questions,
        container_col=SCHEMA.CQ_CONTAINER,
        level_col=SCHEMA.CQ_LEVEL,
        only_active=True,
        active_col=SCHEMA.CQ_ACTIVE,
        round_to=2,
    )

    # Load removed containers (if file missing, returns empty set)
    removed_ids = load_removed_container_ids(container_col=SCHEMA.CQ_CONTAINER)

    subs_df = build_substitution_df(
        book_df=datasets.book_questions,
        cq_df=datasets.containers_questions,
        book_question_id_col=SCHEMA.BOOK_QID,
        book_missing_col=SCHEMA.BOOK_MISSING_FLAG,
        cq_question_id_col=SCHEMA.CQ_QID,
        cq_container_col=SCHEMA.CQ_CONTAINER,
        level_col=SCHEMA.CQ_LEVEL,
        subject_col=SCHEMA.CQ_SUBJECT,
        active_col=SCHEMA.CQ_ACTIVE,
        only_active_candidates=only_active_candidates,
        avg_by_container=avg_by_container,
        removed_container_ids=removed_ids,  # NEW
    )

    subs_df.to_csv(out_csv, index=False, encoding=CSV_WRITE_ENCODING)
    return subs_df, avg_by_container


# =======================
# ====== __main__   =====
# =======================

def main() -> None:
    """
    CLI-like entrypoint. Uses default filenames (CWD, with a fallback to /mnt/data),
    runs the pipeline, and prints a compact summary.
    """
    book_csv = resolve_path(FILE_BOOK)
    containers_questions_csv = resolve_path(FILE_CONTAINERS_QUESTIONS)
    assignments_containers_csv = resolve_path(FILE_ASSIGNMENTS_CONTAINERS)
    out_csv = Path(OUTPUT_CSV)

    subs_df, _avg = run_pipeline(
        book_csv=book_csv,
        containers_questions_csv=containers_questions_csv,
        assignments_containers_csv=assignments_containers_csv,
        out_csv=out_csv,
        only_active_candidates=True,
    )

    removed_ids = load_removed_container_ids(container_col=SCHEMA.CQ_CONTAINER)
    print(f"Removed containers loaded: {len(removed_ids)} (e.g., {next(iter(removed_ids), '—')})")

    print(f"Substitutions created: {len(subs_df)}")
    print(f"Saved to: {out_csv.resolve()} (encoding={CSV_WRITE_ENCODING})")


if __name__ == "__main__":
    main()
