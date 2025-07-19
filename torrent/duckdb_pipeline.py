import os

import duckdb
import pandas as pd

os.makedirs("data/filtered", exist_ok=True)
subreddit = "aiwars"
POSTS_PATH = f"data/extracted/{subreddit}.jsonl"
KEYWORDS_CSV = f"torrent/bias_keywords.csv"
SUBREDDIT_GROUPS_CSV = f"torrent/subreddit_groups.csv"
OUTPUT_PATH = f"data/filtered/{subreddit}_full_filtered_posts.csv"


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


def load_posts(subreddit, path=POSTS_PATH):
    df_posts = pd.read_json(path.format(subreddit=subreddit), lines=True)
    print(df_posts.shape)
    print(df_posts.head())
    # print(df_posts["created_utc"].min(), df_posts["created_utc"].max())
    # print(df_posts["created_utc"].describe())

    return df_posts


def connect_duckdb(db_path="reddit.duckdb"):
    return duckdb.connect(db_path)


def register_tables(conn, df_posts, keywords_csv, subreddit_groups_csv):
    conn.register("df_posts", df_posts)
    print("the number of rows in df_posts")
    print(len(df_posts))

    print(df_posts[["title", "selftext"]].isna().sum())

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


def create_post_views(conn):
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


def create_filtered_view(conn, ai_keywords):
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

    # 2. ai_keywords 조건을 SQL WHERE 절로 생성
    keyword_condition = " OR ".join(
        [f"p.clean_text LIKE '%{kw.lower()}%'" for kw in ai_keywords]
    )

    # 3. filtered_posts 뷰 생성 (bias keyword 포함 조건 추가)
    filtered_posts_view = f"""
    CREATE OR REPLACE VIEW filtered_posts AS
    SELECT *
    FROM posts_with_group_and_keywords p
    WHERE
        ({keyword_condition})
        AND ARRAY_LENGTH(matched_bias_types) > 0
        AND CASE
            WHEN subreddit_group = 'casual' THEN ARRAY_LENGTH(matched_keywords) >= 2
            WHEN subreddit_group = 'expert' THEN ARRAY_LENGTH(matched_keywords) >= 1
            ELSE FALSE
        END;
    """
    conn.execute(filtered_posts_view)


def export_filtered_posts(conn, subreddit, output_path=OUTPUT_PATH):
    df_filtered = conn.execute("SELECT * FROM filtered_posts").df()
    df_filtered.to_csv(output_path.format(subreddit=subreddit), index=False)
    return df_filtered


def statistics(conn, df_filtered):
    print("\n📊 🔎 Analysis Summary ========================")

    # ✅ Total posts
    total_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    print(f"Total posts: {total_posts:,}")

    # ✅ Filtered posts
    filtered_posts = len(df_filtered)
    print(f"Filtered posts: {filtered_posts:,}")

    # ✅ Filtered percentage
    percentage = (filtered_posts / total_posts) * 100
    print(f"Filtered percentage: {percentage:.2f}%")

    # ✅ subreddit_group distribution
    print("\n📊 subreddit_group distribution:")
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
    print("\n📌 Most frequent bias types:")
    print(
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
    print("\n📌 Most frequent keywords:")
    print(
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


def main(subreddit):
    os.makedirs("data/filtered", exist_ok=True)
    df_posts = load_posts(subreddit)
    conn = connect_duckdb()
    register_tables(conn, df_posts, KEYWORDS_CSV, SUBREDDIT_GROUPS_CSV)
    create_post_views(conn)
    create_filtered_view(conn, ai_keywords)

    # 🔍 Debugging SQL log
    print("\n🔍 Check the joined subreddit and group:")
    print(
        conn.execute(
            """
        SELECT subreddit, subreddit_group
        FROM posts_with_group_and_keywords
        ORDER BY subreddit_group
        LIMIT 20;
    """
        ).fetchdf()
    )

    df = export_filtered_posts(conn, subreddit)
    statistics(conn, df)
    print(df.head(10))
    conn.close()


if __name__ == "__main__":
    main("aiwars")
    # test_filtered_view_sample()
"""
    ✅ Save intermediate results (duckdb file)
duckdb.query("CREATE TABLE mytable AS SELECT * FROM 'filtered.jsonl'")
duckdb.query("EXPORT DATABASE 'my_analysis.duckdb' (FORMAT PARQUET);")

"""


def load_sample_posts(path=POSTS_PATH, n=10):
    df_posts = pd.read_json(path, lines=True)
    df_sample = df_posts.head(n)
    print(df_sample[["title", "selftext"]])
    return df_sample


def test_filtered_view_sample():
    df_sample = load_sample_posts(n=10)
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
    print("\n🎯 Sample Test Result:")
    print(df_result)

    conn.close()
