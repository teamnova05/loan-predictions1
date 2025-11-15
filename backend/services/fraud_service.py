from models.model_loader import ModelLoader

class FraudService:
    def __init__(self):
        self.loader = ModelLoader(fraud=True)

    def detect(self, data: dict):
        if not isinstance(data, dict):
            raise ValueError("input should be a JSON object/dict")
        return self.loader.predict_fraud(data)
