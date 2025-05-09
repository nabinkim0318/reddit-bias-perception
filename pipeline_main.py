import time

from processing.clean_text import main as clean_text
from processing.keyword_filter import main as keyword_filter
from processing.llm_few_shot import main as llm_filter
from reddit_crawler.main import main as crawl_reddit


def timed_step(label, func):
    print(f"\n🚩 [{label}] started...")
    start = time.time()
    func()
    end = time.time()
    print(f"✅ [{label}] completed in {end - start:.2f} seconds")


def main():
    timed_step("1. Reddit Crawling", crawl_reddit)
    timed_step("2. Text Cleaning", clean_text)
    timed_step("3. LLM Filtering", llm_filter)
    timed_step("4. Keyword Filtering", keyword_filter)

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
