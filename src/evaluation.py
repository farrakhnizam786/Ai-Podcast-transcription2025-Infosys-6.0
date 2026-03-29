def evaluate_segments(segments):
    return {
        "num_topics": len(segments),
        "avg_length": sum(len(s.split()) for s in segments) / max(len(segments),1)
    }
