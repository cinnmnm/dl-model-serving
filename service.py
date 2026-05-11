import bentoml
import torch
from torchvision import transforms
import PIL.ImageOps
import torch.serialization

mnist_model = bentoml.models.BentoModel("mnist_model:latest")

@bentoml.service(
    name="mnist_classifier",
    resources={"cpu": "1"},
)
class MNISTService:
    model_ref = mnist_model

    def __init__(self):
        torch.serialization.add_safe_globals([torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]) 
        self.model = self.model_ref.load_model(weights_only=False)
        self.model.eval()

    @bentoml.api
    def predict(self, img: "PIL.Image.Image") -> str:
        if img.mode != 'L':
            img = img.convert('L')
        img = PIL.ImageOps.invert(img)
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
            prediction = output.argmax(dim=1).item()
        
        return f"The model predicts this number is: {prediction}\n"