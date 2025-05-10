"""
Topic Modeling Visualizations for AI Bias Reddit Dataset
Includes:
- Barcharts
- Heatmaps
- WordClouds
- Co-occurrence graphs
- Confidence histogram
- Length distribution
- Keyword frequency
"""

from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from bertopic import BERTopic
from wordcloud import WordCloud


def visualize_barchart(model: BERTopic, topics=None, title_suffix=""):
    fig = model.visualize_barchart(topics=topics)
    fig.update_layout(title=f"Topic Frequency {title_suffix}")
    fig.show()


def visualize_heatmap(model: BERTopic, topics, custom_labels=None):
    fig = model.visualize_heatmap(topics=topics)
    if custom_labels:
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(topics))),
                ticktext=[custom_labels[t] for t in topics],
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(topics))),
                ticktext=[custom_labels[t] for t in topics],
            ),
        )
    fig.show()


def visualize_documents(model: BERTopic, docs, embeddings, topics):
    fig = model.visualize_documents(docs, embeddings=embeddings, topics=topics)
    fig.show()


def visualize_wordclouds(model: BERTopic, topics):
    for topic_id in topics:
        words = model.get_topic(topic_id)
        freq = {word: weight for word, weight in words}
        wc = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {topic_id}")
        plt.show()


def visualize_cooccurrence_graph(model: BERTopic, topics):
    for topic_id in topics:
        words = model.get_topic(topic_id)
        G = nx.Graph()
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                G.add_edge(words[i][0], words[j][0])
        plt.figure(figsize=(5, 5))
        nx.draw(
            G, with_labels=True, node_color="skyblue", node_size=1500, edge_color="gray"
        )
        plt.title(f"Topic {topic_id} Co-occurrence")
        plt.show()


def visualize_length_distribution(docs, topics):
    lengths = [len(doc.split()) for doc in docs]
    df = pd.DataFrame({"Topic": topics, "Length": lengths})
    sns.boxplot(x="Topic", y="Length", data=df)
    plt.title("Document Length by Topic")
    plt.show()


def visualize_length_by_label(docs, topics, label_map):
    lengths = [len(doc.split()) for doc in docs]
    df = pd.DataFrame({"Topic": topics, "Length": lengths})
    df["Label"] = df["Topic"].map(label_map)
    df_filtered = df[df["Label"].notnull()]
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="Label", y="Length", data=df_filtered)
    plt.title("Length by Bias Category")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def visualize_confidence(probs):
    max_probs = [max(p) for p in probs]
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=30, color="skyblue", edgecolor="white")
    plt.axvspan(0.0, 0.3, color="red", alpha=0.1, label="Low confidence")
    plt.axvspan(0.9, 1.0, color="green", alpha=0.1, label="High confidence")
    plt.xlabel("Max Topic Probability")
    plt.ylabel("Document Count")
    plt.title("Confidence Distribution of Topic Assignments")
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_keyword_counts(keyword_lists):
    all_keywords = []
    for kw_list in keyword_lists:
        all_keywords.extend(kw_list)
    keyword_counts = Counter(all_keywords)
    df = pd.DataFrame(keyword_counts.items(), columns=["Keyword", "Count"])
    df = df.sort_values(by="Count", ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Keyword", y="Count")
    plt.title("Frequency of Matched Bias Keywords")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
