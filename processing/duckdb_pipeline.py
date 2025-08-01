# processing/duckdb_pipeline.py
import logging
import os
from typing import List

import duckdb
import pandas as pd

from config.config import BASE_DIR
from processing.duckdb_data_processing import decompress_zstd

logging.basicConfig(level=logging.INFO)


KEYWORDS_CSV = f"{BASE_DIR}/config/bias_keywords.csv"
SUBREDDIT_GROUPS_CSV = f"{BASE_DIR}/config/subreddit_groups.csv"
DB_PATH = f"{BASE_DIR}/reddit.duckdb"


def get_paths(subreddit: str):
    return {
        "compressed_path": f"{BASE_DIR}/extracted/{subreddit}_submissions.zst",
        "extracted_path": f"{BASE_DIR}/extracted/{subreddit}_posts.jsonl",
    }


ai_keywords = [
    "ai",
    "artificial intelligence",
    "ai art",
    "ai image",
    "ai image generated",
    "ai pictures",
    "ai photos",
    "ai content",
    "prompt",
    "prompting",
    "ai filter",
    "deepfake",
    "face swap",
    "diffusion",
    "deep learning",
    "machine learning",
    "neural network",
    "stable diffusion",
    "dalle",
    "midjourney",
    "openai",
    "text-to-image",
    "image generation",
    "chatgpt",
    "gpt",
    "llm",
    "copilot",
    "gemini",
]


def load_posts(path: str) -> pd.DataFrame:
    logging.info(f"Loading posts from {path}...")
    try:
        df_posts = pd.read_json(path, lines=True)
        logging.info(df_posts.shape)
        logging.info(df_posts.head())
        logging.info(f"Loaded {len(df_posts)} posts")
    except Exception as e:
        logging.error(f"Compressed file not found: {e}")
        raise

    return df_posts


def connect_duckdb(db_path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path)


def register_tables(
    conn: duckdb.DuckDBPyConnection,
    df_posts: pd.DataFrame,
    keywords_csv: str,
    subreddit_groups_csv: str,
):
    conn.register("df_posts", df_posts)
    logging.info("the number of rows in df_posts")
    logging.info(len(df_posts))

    logging.info(df_posts[["title", "selftext"]].isna().sum())

    # print("the number of nulls in title and selftext")
    # print(df_posts[["title", "selftext"]].applymap(lambda x: str(x).strip().lower()).value_counts())

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE keywords AS
        SELECT * FROM read_csv_auto('{keywords_csv}');
    """
    )

    conn.execute(
        f"""
        CREATE OR REPLACE TABLE subreddit_groups AS
        SELECT * FROM read_csv_auto('{subreddit_groups_csv}');
    """
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
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                REGEXP_REPLACE(
                    LOWER(COALESCE(title, '') || ' ' || COALESCE(selftext, '')),
                    'http\\S+|www\\.\\S+', ''
                ),
                '<.*?>', ''
            ),
            '[^a-z\\s]', ''
        ) AS clean_text,
        COALESCE(title, '') || ' ' || COALESCE(selftext, '') AS full_text
    FROM df_posts
    WHERE
        created_utc BETWEEN 1640995200 AND 1735689599
        AND (
            LOWER(COALESCE(title, '')) NOT IN ('[deleted]', '[removed]', '', 'null', 'none')
            OR LOWER(COALESCE(selftext, '')) NOT IN ('[deleted]', '[removed]', '', 'null', 'none')
        );
    """
    )


def create_filtered_view(conn: duckdb.DuckDBPyConnection, ai_keywords: List[str]):
    # 1. 먼저 posts_with_group_and_keywords 뷰 생성
    posts_with_keywords_view = """
    CREATE OR REPLACE VIEW posts_with_group_and_keywords AS
    SELECT
        p.id,
        p.subreddit,
        sg.group AS subreddit_group,
        p.clean_text,
        p.full_text,
        ARRAY_AGG(DISTINCT k.category) AS matched_bias_types,
        ARRAY_AGG(DISTINCT k.keyword) AS matched_keywords
    FROM posts p
    LEFT JOIN subreddit_groups sg
        ON LOWER(p.subreddit) = LOWER(sg.subreddit)
    JOIN keywords k
        ON p.clean_text LIKE '%' || k.keyword || '%'
    GROUP BY p.id, p.subreddit, p.clean_text, p.full_text, sg.group;
    """
    conn.execute(posts_with_keywords_view)

    # 2. automatically generate ai_keywords condition as SQL WHERE clause
    keyword_condition = " OR ".join(
        [f"p.clean_text LIKE '%{kw.lower()}%'" for kw in ai_keywords]
    )

    # 3. create filtered_posts view (add bias keyword condition)
    filtered_posts_view = f"""
    CREATE OR REPLACE VIEW filtered_posts AS
    SELECT *
    FROM posts_with_group_and_keywords p
    WHERE
        ({keyword_condition})
        AND ARRAY_LENGTH(matched_bias_types) > 0
        AND CASE
            WHEN LOWER(subreddit) = 'twoxchromosomes' THEN ARRAY_LENGTH(matched_keywords) >= 2

            WHEN subreddit_group = 'technical' THEN ARRAY_LENGTH(matched_bias_types) > 0

            WHEN subreddit_group = 'creative_AI_communities' THEN ARRAY_LENGTH(matched_bias_types) > 0

            WHEN subreddit_group = 'critical_discussion' THEN ARRAY_LENGTH(matched_bias_types) > 0

            WHEN subreddit_group = 'general_reddit' THEN ARRAY_LENGTH(matched_keywords) >= 2

            ELSE FALSE
        END;
    """
    conn.execute(filtered_posts_view)


def export_filtered_posts(
    conn: duckdb.DuckDBPyConnection, output_path: str
) -> pd.DataFrame:
    df_filtered = conn.execute("SELECT * FROM filtered_posts").df()
    df_filtered.to_csv(output_path, index=False)
    return df_filtered


def statistics(conn: duckdb.DuckDBPyConnection, df_filtered: pd.DataFrame):
    logging.info("\n📊 🔎 Analysis Summary ========================")

    # ✅ Total posts
    result = conn.execute("SELECT COUNT(*) FROM posts").fetchone()
    total_posts = result[0] if result else 0
    logging.info(f"Total posts: {total_posts:,}")

    # ✅ Filtered posts
    filtered_posts = len(df_filtered)
    logging.info(f"Filtered posts: {filtered_posts:,}")

    # ✅ Filtered percentage
    percentage = (filtered_posts / total_posts) * 100
    logging.info(f"Filtered percentage: {percentage:.2f}%")

    # ✅ subreddit_group distribution
    logging.info("\n📊 subreddit_group distribution:")
    print(
        conn.execute(
            """
        SELECT subreddit_group, COUNT(*) AS count
        FROM filtered_posts
        GROUP BY subreddit_group
        ORDER BY count DESC;
    """
        ).fetchdf()
    )

    # ✅ Most frequent bias types
    logging.info("\n📌 Most frequent bias types:")
    logging.info(
        conn.execute(
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
        ).fetchdf()
    )

    # ✅ Most frequent keywords
    logging.info("\n📌 Most frequent keywords:")
    logging.info(
        conn.execute(
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
        ).fetchdf()
    )


def main(subreddit: str):
    os.makedirs(f"{BASE_DIR}/processed/filtered", exist_ok=True)
    os.makedirs(
        os.path.dirname(f"{BASE_DIR}/extracted/{subreddit}.jsonl"), exist_ok=True
    )

    paths = get_paths(subreddit)

    # decompress zstd
    if not os.path.exists(paths["extracted_path"]):
        decompress_zstd(
            paths, prefer_cli=True
        )

    # load posts
    df_posts = load_posts(paths["extracted_path"])
    conn = connect_duckdb()
    register_tables(conn, df_posts, KEYWORDS_CSV, SUBREDDIT_GROUPS_CSV)
    create_post_views(conn)
    create_filtered_view(conn, ai_keywords)

    # Debugging SQL log
    logging.info("\n🔍 Check the joined subreddit and group:")
    logging.info(
        conn.execute(
            """
        SELECT subreddit, subreddit_group
        FROM posts_with_group_and_keywords
        ORDER BY subreddit_group
        LIMIT 20;
    """
        ).fetchdf()
    )

    # export filtered posts
    df = export_filtered_posts(
        conn,
        f"{BASE_DIR}/processed/filtered/{subreddit}_full_filtered_posts.csv",
    )
    statistics(conn, df)
    logging.info(df.head(10))
    conn.close()


if __name__ == "__main__":
    subreddit = "aiwars"
    main(subreddit)
    # test_filtered_view_sample(subreddit)
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
    register_tables(conn, df_sample, KEYWORDS_CSV, SUBREDDIT_GROUPS_CSV)

    # Create views
    create_post_views(conn)
    create_filtered_view(conn, ai_keywords)

    # Check results
    df_result = conn.execute("SELECT * FROM filtered_posts").df()
    logging.info("\n🎯 Sample Test Result:")
    logging.info(df_result)

    conn.close()
