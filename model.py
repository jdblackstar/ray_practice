from transformers import pipeline


class SentimentAnalysisModel:
    def __init__(self, model_name, model_version):
        self.model = pipeline(
            "sentiment-analysis", model=model_name, revision=model_version
        )

    def __call__(self, text):
        return self.model(text)[0]
