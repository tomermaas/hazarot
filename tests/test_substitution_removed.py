import pandas as pd
from substitutions import substitutions as sub


def test_build_substitution_df_skips_removed_containers():
    # book has two questions, one in removed container 200, one in 300
    book_df = pd.DataFrame({
        sub.SCHEMA.BOOK_QID: ["C1", "B2"],
        sub.SCHEMA.BOOK_MISSING_FLAG: [0, 0],
    })

    cq_df = pd.DataFrame([
        # assigned rows (map old qids to container/subject/level)
        ["C1", 200, 3, "math", 1],   # removed container -> should be skipped entirely
        ["B2", 300, 3, "math", 1],   # will be substituted
        # candidates
        ["X1", 0,   3, "math", 1],
        ["X2", 0,   2, "science", 1],
    ], columns=[
        sub.SCHEMA.CQ_QID, sub.SCHEMA.CQ_CONTAINER, sub.SCHEMA.CQ_LEVEL,
        sub.SCHEMA.CQ_SUBJECT, sub.SCHEMA.CQ_ACTIVE
    ])

    df = sub.build_substitution_df(
        book_df=book_df,
        cq_df=cq_df,
        removed_container_ids={"200"},
        avg_by_container=None,
        only_active_candidates=True,
    )

    # Expect only one row for container 300 and candidate X1
    assert len(df) == 1
    assert df.iloc[0]["container_id"] == "300"
    assert df.iloc[0]["new_question_id"] == "X1"
