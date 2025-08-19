import re
import pandas as pd

# ─── Helper Functions ─────────────────────────────────────────────────────────

def normalize_old_id(s: str) -> str:
    """
    Remove RTL/LTR control characters and normalize all dash variants
    to ASCII hyphen-minus (-).
    """
    s = re.sub(r'[\u200E\u200F]', '', s)
    s = re.sub(r'[\u2010-\u2015]', '-', s)
    return s

def parse_page_question_by_whitespace(raw: str):
    """
    Given any old_question_id string, return (page, question) as ints.
    - If raw contains whitespace, treat first number as page, second as question.
    - Otherwise, first number is question, second is page.
    Returns (None, None) if fewer than two numbers are found.
    """
    norm = normalize_old_id(str(raw))
    nums = re.findall(r'\d+', norm)
    if len(nums) < 2:
        return None, None
    if re.search(r'\s', raw):
        return int(nums[0]), int(nums[1])
    else:
        return int(nums[1]), int(nums[0])

def find_old_question_col(columns):
    """
    Locate the column name for old_question_id, even if it's slightly different.
    """
    for col in columns:
        if col.strip().lower() == 'old_question_id':
            return col
    candidates = [col for col in columns
                  if 'old' in col.lower()
                  and 'question' in col.lower()
                  and 'id' in col.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(f"No 'old_question_id' column in {list(columns)}")

def load_raw_reports(paths):
    """
    Read all CSVs, rename their old_question_id column to a uniform name,
    and concatenate into one DataFrame.
    """
    dfs = []
    for path in paths:
        df = pd.read_csv(path, encoding='utf-8-sig')
        old_col = find_old_question_col(df.columns)
        df = df.rename(columns={old_col: 'old_question_id'})
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def add_page_question_columns(df):
    """
    Add integer columns 'page_number' and 'question_number' to df
    by parsing the raw old_question_id values.
    """
    pq = df['old_question_id'].astype(str).apply(
        lambda raw: pd.Series(
            parse_page_question_by_whitespace(raw),
            index=['page_number','question_number']
        )
    )
    return pd.concat([df, pq], axis=1)

# ─── Core Processing ──────────────────────────────────────────────────────────

def process_reports(paths, export_path):
    """
    Load CSVs at `paths`, keep only codes starting with 'סח-',
    parse page/question, dedupe & sort, and export the five-column CSV.
    """
    df = load_raw_reports(paths)

    # Keep only rows where old_question_id starts with 'סח-'
    df = df[df['old_question_id']
              .astype(str)
              .str.match(r'^סח\s*-')].copy()

    df = add_page_question_columns(df)

    # Drop rows that didn't parse
    df = df.dropna(subset=['page_number','question_number'])

    # Cast to int, dedupe, sort
    df = (
        df.assign(
            page_number=lambda d: d['page_number'].astype(int),
            question_number=lambda d: d['question_number'].astype(int)
        )
        .drop_duplicates(subset=['question_id'])
        .sort_values(by=['page_number','question_number'])
    )

    # Select only the five required columns
    final = df[['question_id','main_subject','difficulty_level','page_number','question_number']]

    # Export as UTF-8 with BOM for Hebrew support
    final.to_csv(export_path, index=False, encoding='utf-8-sig')
    print(f"Exported {len(final)} questions to {export_path}")
    return final

# ─── Lookup Function ──────────────────────────────────────────────────────────

def find_question_ids(page, question, paths=None, raw_df=None):
    """
    Search raw reports for all question_id values at the given page & question.
    Supply either `raw_df` or `paths` (list of CSV paths).
    """
    if raw_df is None:
        if paths is None:
            raise ValueError("Provide either raw_df or paths to load raw reports.")
        raw_df = load_raw_reports(paths)

    raw_df = add_page_question_columns(raw_df)
    matches = raw_df.loc[
        (raw_df['page_number'] == page) &
        (raw_df['question_number'] == question),
        'question_id'
    ].tolist()

    if matches:
        print(f"Question IDs on page {page}, question {question}: {matches}")
    else:
        print(f"No matches found for page {page}, question {question}.")
    return matches

# ─── Script Entry Point ──────────────────────────────────────────────────────

if __name__ == '__main__':
    # For processed_questions.csv we only use the first two reports:
    REPORT_PATHS = [
        "Questions_Reports_Task_Questions.csv",
        "Questions_Reports_None_Task_Questions.csv"
    ]
    OUTPUT_CSV = "processed_questions.csv"

    # Process only סח- codes
    #processed_df = process_reports(REPORT_PATHS, OUTPUT_CSV)

    # If you need to lookup in all raw reports (including missing file), use:
    ALL_PATHS = REPORT_PATHS + ["Questions_Reports_Missing_Question.csv"]
    find_question_ids(65, 6, paths=ALL_PATHS)
