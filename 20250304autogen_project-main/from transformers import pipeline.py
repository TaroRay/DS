from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face!")
print(result)