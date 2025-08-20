"""
CLI entry-point for hazarot-subst.

Usage:
  haz-subst --book book_qusetions.csv --containers-questions containers_questions.csv             [--removed containers_to_be_removed.csv] [--out substitution_plan.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .substitutions import (
    SCHEMA,
    CSV_WRITE_ENCODING,
    read_csv,
    resolve_path,
    average_question_level_by_container,
    build_substitution_df,
    load_removed_container_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="haz-subst", description="Build substitution plan CSV.")
    p.add_argument("--book", dest="book_csv", default="book_qusetions.csv", help="Path to book_qusetions.csv")
    p.add_argument("--containers-questions", dest="cq_csv", default="containers_questions.csv", help="Path to containers_questions.csv")
    p.add_argument("--assignments-containers", dest="ac_csv", default="assignments_containers.csv", help="(Optional) Path to assignments_containers.csv (kept for parity)")
    p.add_argument("--removed", dest="removed_csv", default=None, help="Path to containers_to_be_removed.csv (optional)")
    p.add_argument("--out", dest="out_csv", default="substitution_plan.csv", help="Output CSV path (default: substitution_plan.csv)")
    p.add_argument("--include-inactive", action="store_true", help="Include inactive questions in candidate pool (default: only active)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    book_csv = resolve_path(args.book_csv)
    cq_csv = resolve_path(args.cq_csv)
    # assignments file isn't required by the builder, but we keep parity with your original pipeline
    _ac_csv = resolve_path(args.ac_csv)

    book_df = read_csv(book_csv)
    cq_df = read_csv(cq_csv)

    avg_by_container = average_question_level_by_container(
        cq_df,
        container_col=SCHEMA.CQ_CONTAINER,
        level_col=SCHEMA.CQ_LEVEL,
        only_active=True,
        active_col=SCHEMA.CQ_ACTIVE,
        round_to=2,
    )

    # Load removed ids either from explicit flag or default file
    if args.removed_csv is not None:
        removed_ids = load_removed_container_ids(path=Path(args.removed_csv), container_col=SCHEMA.CQ_CONTAINER)
    else:
        removed_ids = load_removed_container_ids(container_col=SCHEMA.CQ_CONTAINER)

    subs_df = build_substitution_df(
        book_df=book_df,
        cq_df=cq_df,
        book_question_id_col=SCHEMA.BOOK_QID,
        book_missing_col=SCHEMA.BOOK_MISSING_FLAG,
        cq_question_id_col=SCHEMA.CQ_QID,
        cq_container_col=SCHEMA.CQ_CONTAINER,
        level_col=SCHEMA.CQ_LEVEL,
        subject_col=SCHEMA.CQ_SUBJECT,
        active_col=SCHEMA.CQ_ACTIVE,
        only_active_candidates=(not args.include_inactive),
        avg_by_container=avg_by_container,
        removed_container_ids=removed_ids,
    )

    out_csv = Path(args.out_csv)
    subs_df.to_csv(out_csv, index=False, encoding=CSV_WRITE_ENCODING)

    print(f"Removed containers loaded: {len(removed_ids or [])}")
    print(f"Substitutions created: {len(subs_df)}")
    print(f"Saved to: {out_csv.resolve()} (encoding={CSV_WRITE_ENCODING})")


if __name__ == "__main__":
    main()
