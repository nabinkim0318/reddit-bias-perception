import os

import pandas as pd

import duckdb

os.makedirs("data/filtered", exist_ok=True)
POSTS_PATH = "data/raw/askreddit.jsonl"
KEYWORDS_CSV = "duckdb/bias_keywords.csv"
SUBREDDIT_GROUPS_CSV = "duckdb/subreddit_groups.csv"
OUTPUT_PATH = "data/filtered/full_filtered_posts.csv"


def load_posts(path=POSTS_PATH):
    return pd.read_json(path, lines=True)


def connect_duckdb(db_path="reddit.duckdb"):
    return duckdb.connect(db_path)


def register_tables(conn, df_posts, keywords_csv, subreddit_groups_csv):
    conn.register("df_posts", df_posts)

    conn.execute(
        f"""
    CREATE OR REPLACE TABLE keywords AS
    SELECT * FROM '{keywords_csv}' (AUTO_DETECT TRUE);
    """
    )

    conn.execute(
        f"""
    CREATE OR REPLACE TABLE subreddit_groups AS
    SELECT * FROM '{subreddit_groups_csv}' (AUTO_DETECT TRUE);
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


def create_filtered_view(conn):
    conn.execute(
        """
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
    )

    # 이 아래에서 조건 분기 + 필터링된 뷰를 추가로 만들어줌
    conn.execute(
        """
    CREATE OR REPLACE VIEW filtered_posts AS
    SELECT *
    FROM posts_with_group_and_keywords
    WHERE
        CASE
            WHEN subreddit_group = 'casual' THEN ARRAY_LENGTH(matched_keywords) >= 2
            WHEN subreddit_group = 'expert' THEN ARRAY_LENGTH(matched_keywords) >= 1
            ELSE FALSE
        END;
    """
    )


def export_filtered_posts(conn, output_path=OUTPUT_PATH):
    df_filtered = conn.execute("SELECT * FROM filtered_posts").df()
    df_filtered.to_csv(output_path, index=False)
    return df_filtered


def main():
    os.makedirs("data/filtered", exist_ok=True)
    df_posts = load_posts()
    conn = connect_duckdb()
    register_tables(conn, df_posts, KEYWORDS_CSV, SUBREDDIT_GROUPS_CSV)
    create_post_views(conn)
    create_filtered_view(conn)
    df = export_filtered_posts(conn)
    print(df.head(10))
    conn.close()


if __name__ == "__main__":
    main()
"""
    ✅ 중간 결과 저장 (duckdb 파일로)
python
Copy
Edit
duckdb.query("CREATE TABLE mytable AS SELECT * FROM 'filtered.jsonl'")
duckdb.query("EXPORT DATABASE 'my_analysis.duckdb' (FORMAT PARQUET);")

"""
