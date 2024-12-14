from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="./code_english_model/", device="cuda:0")
text = "This nice, I love you. Can you fix it?"
print(text, classifier(text))
text = "<html><head><title>Test</title></head><body><p>Hello World</p></body></html>"
print(text, classifier(text))
