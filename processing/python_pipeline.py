# processing/python_pipeline.py
import ast
import logging
from pathlib import Path

import duckdb
import pandas as pd

from config.config import BASE_DIR, CONFIG_DIR
from processing.duckdb_data_processing import decompress_zstd
from config.config import AI_KEYWORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---- Config paths ----
BASE_DIR = Path(BASE_DIR)
KEYWORDS_CSV = CONFIG_DIR / "bias_keywords.csv"
SUBREDDIT_GROUPS_CSV = CONFIG_DIR / "subreddit_groups.csv"
DB_PATH = BASE_DIR / "reddit.duckdb"



# ---- Functions ----
def get_paths(subreddit: str):
    return {
        "compressed_path": Path(BASE_DIR)
        / "extracted"
        / f"{subreddit}_submissions.zst",
        "extracted_path": Path(BASE_DIR) / "extracted" / f"{subreddit}_posts.jsonl",
    }


def load_posts(path: str) -> pd.DataFrame:
    logging.info(f"📄 Loading posts from {path}...")
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ JSONL not found: {path}")

    df_posts = pd.read_json(path, lines=True)
    logging.info(
        f"✅ Loaded {len(df_posts)} posts with {len(df_posts.columns)} columns"
    )
    return df_posts


# ---- DuckDB
def connect(in_memory: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:" if in_memory else str(DB_PATH))


def register_tables(conn: duckdb.DuckDBPyConnection, df_posts: pd.DataFrame):
    conn.register("df_posts", df_posts)
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE keywords AS
        SELECT * FROM read_csv_auto('{KEYWORDS_CSV}');
    """
    )
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE subreddit_groups AS
        SELECT * FROM read_csv_auto('{SUBREDDIT_GROUPS_CSV}');
    """
    )
    # ✅ AI keywords tentative table
    ai_vals = []
    for kw in AI_KEYWORDS:
        escaped_kw = kw.lower().replace("'", "''")
        ai_vals.append(f"('{escaped_kw}')")
    ai_vals_str = ", ".join(ai_vals)
    conn.execute(
        f"CREATE OR REPLACE TEMP TABLE ai_kw(keyword) AS VALUES {ai_vals_str};"
    )


def create_post_views(conn: duckdb.DuckDBPyConnection):
    conn.execute(
        """
        CREATE OR REPLACE VIEW posts AS
        SELECT
            id,
            subreddit,
            title,
            selftext,
            created_utc,

            -- 1) LLM input: URL/HTML only remove (preserve case/punctuation/emojis)
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    COALESCE(title, '') || ' ' || COALESCE(selftext, ''),
                    'http\\S+|www\\.\\S+', ''
                ),
                '<.*?>', ''
            ) AS clean_text,

            -- 2) Matching use: lowercase + URL/HTML remove + prompt symbols(: - / _ %) preserve
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    REGEXP_REPLACE(
                        LOWER(COALESCE(title, '') || ' ' || COALESCE(selftext, '')),
                        'http\\S+|www\\.\\S+', ''
                    ),
                    '<.*?>', ''
                ),
                '[^a-z0-9\\s:\\-/_%'
                || '\\x{1F300}-\\x{1FAFF}'
                || '\\x{2600}-\\x{26FF}'
                || '\\x{1F3FB}-\\x{1F3FF}'  -- skin tone
                || '\\x{200D}'              -- ZWJ
                || '\\x{FE0F}'              -- VS16
                || ']',
                ' '
            ) AS clean_text_lc,

         
            COALESCE(title, '') || ' ' || COALESCE(selftext, '') AS full_text
        FROM df_posts
        WHERE
            -- 2022-01-01 ~ 2024-12-31
            created_utc BETWEEN 1640995200 AND 1735689599
            AND (
                LOWER(COALESCE(title, '')) NOT IN ('[deleted]', '[removed]', '', 'null', 'none')
                OR LOWER(COALESCE(selftext, '')) NOT IN ('[deleted]', '[removed]', '', 'null', 'none')
            );
        """
    )
    logging.info("✅ Created posts view")


def create_filtered_views(conn: duckdb.DuckDBPyConnection):
    # 1) Keywords: regex special char escape view
    conn.execute(
        """
        CREATE OR REPLACE VIEW keywords_norm AS
        SELECT
            TRIM(LOWER(keyword))  AS keyword,
            TRIM(LOWER(category)) AS category,
            REGEXP_REPLACE(TRIM(LOWER(keyword)), '([.^$|()*+?\\[\\]{}\\\\])', '\\\\$1') AS kw_esc
        FROM keywords;
    """
    )

    # 2) posts_with_group_and_keywords: REGEXP_MATCH for word boundary matching
    conn.execute(
        """
        CREATE OR REPLACE VIEW posts_with_group_and_keywords AS
        SELECT
            p.id,
            p.subreddit,
            sg."group" AS subreddit_group,
            p.clean_text,      -- LLM use (preserve original text)
            p.clean_text_lc,   -- matching use (lowercase/normalized)
            p.full_text,
            ARRAY_AGG(DISTINCT k.category) AS matched_bias_types,
            ARRAY_AGG(DISTINCT k.keyword)  AS matched_keywords
        FROM posts p
        LEFT JOIN subreddit_groups sg
            ON LOWER(p.subreddit) = LOWER(sg.subreddit)
        JOIN keywords_norm k
        ON REGEXP_MATCHES(' ' || p.clean_text_lc || ' ', '(^|\\s)' || k.kw_esc || '(\\s|$)')
        GROUP BY p.id, p.subreddit, p.clean_text, p.clean_text_lc, p.full_text, sg."group";
    """
    )

    # 3) AI keywords: REGEXP_MATCH (simple version: OR connected)
    conn.execute(
        """
        CREATE OR REPLACE VIEW filtered_posts AS
        SELECT *
        FROM posts_with_group_and_keywords p
        WHERE
            EXISTS (
                SELECT 1
                FROM ai_kw a
                WHERE REGEXP_MATCHES(
                    ' ' || p.clean_text_lc || ' ',
                    '(^|\\s)' || REGEXP_REPLACE(a.keyword, '([.^$|()*+?\\[\\]{}\\\\])','\\\\$1') || '(?:es|s|ed|al|ical|y)?(\\s|$)'
                )
            )
            AND ARRAY_LENGTH(matched_bias_types) > 0
            AND CASE
                WHEN LOWER(subreddit) = 'twoxchromosomes' THEN ARRAY_LENGTH(matched_keywords) >= 2
                WHEN subreddit_group IN ('technical','creative_ai_communities','critical_discussion') THEN ARRAY_LENGTH(matched_bias_types) > 0
                WHEN subreddit_group = 'general_reddit' THEN ARRAY_LENGTH(matched_keywords) >= 2
                ELSE FALSE
            END;
    """
    )
    logging.info("✅ Created filtered_posts view (regex + EXISTS)")


def export_df(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute("SELECT * FROM filtered_posts").df()


def flatten_listlike(x):
    if isinstance(x, list):
        return ", ".join(map(str, x))
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            arr = ast.literal_eval(x)
            if isinstance(arr, list):
                return ", ".join(map(str, arr))
        except Exception:
            return x
    return x


def statistics(conn: duckdb.DuckDBPyConnection, df_filtered: pd.DataFrame):
    logging.info("\n📊 🔎 Analysis Summary ========================")

    # ✅ Total posts count
    result = conn.execute("SELECT COUNT(*) FROM posts").fetchone()
    total_posts = result[0] if result else 0
    logging.info(f"Total posts: {total_posts:,}")

    # ✅ Filtered posts count
    filtered_posts = len(df_filtered)
    logging.info(f"Filtered posts: {filtered_posts:,}")

    # ✅ Filtered posts percentage
    percentage = (filtered_posts / total_posts * 100) if total_posts else 0.0
    logging.info(f"Filtered percentage: {percentage:.2f}%")

    # ✅ subreddit_group distribution count
    logging.info("\n📊 subreddit_group distribution:")
    df = conn.execute(
        """
        SELECT subreddit_group, COUNT(*) AS count
        FROM filtered_posts
        GROUP BY subreddit_group
        ORDER BY count DESC;
    """
    ).df()
    logging.info("\n%s", df.head(20).to_string(index=False))

    # ✅ Most frequent bias types count
    logging.info("\n📌 Most frequent bias types:")
    df = conn.execute(
        """
        SELECT bias_type, COUNT(*) AS count
        FROM (
            SELECT UNNEST(matched_bias_types) AS bias_type
            FROM filtered_posts
        )
        GROUP BY bias_type
        ORDER BY count DESC
        LIMIT 10;
    """
    ).df()
    logging.info("\n%s", df.head(20).to_string(index=False))

    # ✅ Most frequent keywords count
    logging.info("\n📌 Most frequent keywords:")
    df = conn.execute(
        """
        SELECT k.category, k.keyword, COUNT(*) AS count
        FROM (
            SELECT UNNEST(matched_keywords) AS keyword
            FROM filtered_posts
        ) AS u
        JOIN keywords k
        ON LOWER(u.keyword) = LOWER(k.keyword)
        GROUP BY k.category, k.keyword
        ORDER BY count DESC
        LIMIT 20;
    """
    ).df()
    logging.info("\n%s", df.head(20).to_string(index=False))


# ---- Orchestrator
def run(subreddit: str) -> Path:
    paths = get_paths(subreddit)

    # 0) ensure JSONL exists and decompress if not
    if not paths["extracted_path"].exists():
        if not paths["compressed_path"].exists():
            raise FileNotFoundError(f"Missing input: {paths['compressed_path']}")
        paths["extracted_path"].parent.mkdir(parents=True, exist_ok=True)
        decompress_zstd(
            {
                "compressed_path": paths["compressed_path"],
                "extracted_path": paths["extracted_path"],
            },
            prefer_cli=True,
        )

    # 1) JSONL → pandas DataFrame
    df_posts = pd.read_json(str(paths["extracted_path"]), lines=True)
    logging.info(f"Loaded {len(df_posts)} posts for {subreddit}")

    # 2) DuckDB filter: create views
    conn = connect(in_memory=True)
    register_tables(conn, df_posts)
    create_post_views(conn)
    create_filtered_views(conn)
    df = export_df(conn)

    # 3) Python post-processing: stopwords AFTER matching + flatten arrays
    # df["clean_text"] = df["clean_text"].fillna("").astype(str).apply(remove_stopwords)
    for col in ("matched_bias_types", "matched_keywords"):
        if col in df.columns:
            df[col] = df[col].apply(flatten_listlike)

    # 4) save
    out_dir = Path(BASE_DIR) / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{subreddit}_filtered_cleaned.csv"
    df.to_csv(str(out_csv), index=False)

    logging.info(f"✅ Saved: {out_csv}  (rows={len(df)})")
    conn.close()
    return out_csv


if __name__ == "__main__":
    run("aiwars")


"""
    ✅ Save intermediate results (duckdb file)
duckdb.query("CREATE TABLE mytable AS SELECT * FROM 'filtered.jsonl'")
duckdb.query("EXPORT DATABASE 'my_analysis.duckdb' (FORMAT PARQUET);")

"""


def load_sample_posts(path, n=10):
    df_posts = pd.read_json(path, lines=True)
    df_sample = df_posts.head(n)
    logging.info(df_sample[["title", "selftext"]])
    return df_sample


def test_filtered_view_sample(subreddit):
    df_sample = load_sample_posts(f"{BASE_DIR}/extracted/{subreddit}_posts.jsonl", n=10)
    conn = duckdb.connect()

    # Register table
    conn.register("df_posts", df_sample)

    # Register tables
    register_tables(conn, df_sample)

    # Create views
    create_post_views(conn)
    create_filtered_views(conn)

    # Check results
    df_result = conn.execute("SELECT * FROM filtered_posts").df()
    logging.info("\n🎯 Sample Test Result:")
    logging.info(df_result)

    conn.close()


# ---- Main ----
def main(subreddit: str):
    return run(subreddit)
