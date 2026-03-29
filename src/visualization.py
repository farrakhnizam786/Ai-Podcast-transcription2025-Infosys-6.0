import json
import matplotlib.pyplot as plt
from collections import Counter

def sentiment_trend(segments):
    lengths = [len(s.split()) for s in segments]
    plt.plot(lengths)
    plt.title("Topic Length Trend")
    plt.xlabel("Topic Index")
    plt.ylabel("Word Count")
    return plt

def word_cloud_data(text):
    words = text.lower().split()
    return Counter(words)
