# 연결된 상태에서 한 번 실행
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 예시
topic_model = BERTopic(embedding_model=embedding_model)

# 학습 후 저장
topic_model.save("models/bertopic_model")
