import pandas as pd

def test_to_int_like_str_series(sub):
    s = pd.Series(["1", "2.0", "003", "x", None])
    out = sub.to_int_like_str_series(s)
    assert out.tolist()[:4] == ["1", "2", "3", "x"]


def test_normalize_subject_series(sub):
    s = pd.Series(["  Math ", "HeBrew", None])
    out = sub.normalize_subject_series(s)
    assert out.tolist() == ["math", "hebrew", "none"]


def test_active_and_container_masks(sub):
    df = pd.DataFrame({
        sub.SCHEMA.CQ_ACTIVE: [1, "yes", "no", 0, None],
        sub.SCHEMA.CQ_CONTAINER: [1, "2", "0", "", None],
    })
    act = sub.is_active_mask(df[sub.SCHEMA.CQ_ACTIVE])
    assert act.tolist() == [True, True, False, False, False]

    has_mask = sub.has_container_mask(df[sub.SCHEMA.CQ_CONTAINER])
    assert has_mask.tolist() == [True, True, False, False, False]

    no_has_mask = sub.no_container_mask(df[sub.SCHEMA.CQ_CONTAINER])
    assert no_has_mask.tolist() == [False, False, True, True, True]
