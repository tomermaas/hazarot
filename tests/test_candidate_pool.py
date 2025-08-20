import pandas as pd
from substitutions import substitutions as sub


def test_build_candidate_pool_includes_removed_containers_and_excludes_book_ids():
    # Minimal cq_df
    df = pd.DataFrame([
        # qid, container, level, subject, active
        ["A1", 0,   2, "Math", 1],        # unassigned -> candidate
        ["B1", 200, 3, "Math", "yes"],    # assigned to REMOVED -> candidate
        ["C1", 100, 5, "Sci",  1],        # assigned (not removed) -> NOT candidate
        ["D1", 0,   1, "Math", 0],        # unassigned but inactive -> NOT when only_active=True
    ], columns=[
        sub.SCHEMA.CQ_QID, sub.SCHEMA.CQ_CONTAINER, sub.SCHEMA.CQ_LEVEL,
        sub.SCHEMA.CQ_SUBJECT, sub.SCHEMA.CQ_ACTIVE
    ])

    book_qids = ["C1"]  # do not reuse originals
    removed = {"200"}

    pool, median = sub.build_candidate_pool(
        df,
        book_qids=book_qids,
        only_active_candidates=True,
        removed_container_ids=removed,
    )

    got = set(pool[sub.SCHEMA.CQ_QID].tolist())
    assert got == {"A1", "B1"}
    assert isinstance(median, float)
