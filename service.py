import bentoml
import torch
from model import CustomMNISTModel

@bentoml.service
class MNISTService:
    def __init__(self):
        self.model = CustomMNISTModel()
        self.model.load_state_dict(torch.load("model.pth", weights_only=True))
        self.model.eval()

    @bentoml.api
    def predict(self, image_array: list) -> int:
        """
        Accepts a JSON array representing a 28x28 image.
        """
        input_tensor = torch.tensor(image_array, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            prediction = torch.argmax(outputs, dim=1).item()
            
        return prediction