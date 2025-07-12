import json
from collections import Counter

import pandas as pd

# JSON 파일 읽기
with open("data/raw/reddit_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# subreddit 정보 추출
subreddits = [post.get("subreddit", "unknown") for post in data]

# subreddit별 개수 계산
subreddit_counts = Counter(subreddits)

print(f"총 {len(data)}개의 포스트가 있습니다.")
print(f"총 {len(subreddit_counts)}개의 서브레딧에서 가져왔습니다.\n")

print("=== 서브레딧별 포스트 개수 ===")
for subreddit, count in sorted(
    subreddit_counts.items(), key=lambda x: x[1], reverse=True
):
    print(f"r/{subreddit}: {count}개")

print(f"\n=== 요약 ===")
max_subreddit = max(subreddit_counts.items(), key=lambda x: x[1])[0]
min_subreddit = min(subreddit_counts.items(), key=lambda x: x[1])[0]
print(f"가장 많은 포스트: r/{max_subreddit} ({subreddit_counts[max_subreddit]}개)")
print(f"가장 적은 포스트: r/{min_subreddit} ({subreddit_counts[min_subreddit]}개)")
print(f"평균 포스트 수: {sum(subreddit_counts.values()) / len(subreddit_counts):.1f}개")
