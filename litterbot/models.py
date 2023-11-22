import torch
import torchvision
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image

from utils import get_logger


logger = get_logger(__name__, __file__)

class ModelBase:
    def __init__(self, model_path, model_type, pixel_range=1.0, RGB_mean=[0,0,0], RGB_stdev=[0,0,0], device=None):
        device                  = "cuda" if not device and torch.cuda.is_available() else "cpu"

        self.px_range           = pixel_range
        self.device             = torch.device(device)
        self.model_type         = model_type
        self.model              = self._load_model(model_path)
        self.array_mean         = self.px_range * np.array(RGB_mean).reshape(3, 1, 1)
        self.array_stdev        = self.px_range * np.array(RGB_stdev).reshape(3, 1, 1)
        self.tensor_mean        = (torch.tensor(RGB_mean, dtype=torch.float).view(1, 3, 1, 1) * (1.0 / self.px_range)).to(self.device).half()
        self.tensor_stdev       = (torch.tensor(RGB_stdev, dtype=torch.float).view(1, 3, 1, 1) * (1.0 / self.px_range)).to(self.device).half()


        #self.mean       = pixel_range * np.array([RGB_mean])
        #self.stdev      = pixel_range * np.array([RGB_stdev])
        
    def _load_model(self, model_path):
        if self.model_type == "alexnet":
            model = torchvision.models.alexnet(pretrained=False)
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
            model.load_state_dict(torch.load(model_path))
            model = model.to(self.device)

        elif self.model_type == "resnet18":
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(512, 2)
            model.load_state_dict(torch.load(model_path))
            model = model.to(self.device)
            model = model.eval().half()

        else:
            raise NotImplementedError(f"Model {self.model_type} not implemented")
        
        return model

    def preprocess(self, frame):
        if self.model_type == "alexnet":
            normalize = transforms.Normalize(self.array_mean[0] / self.px_range, self.array_stdev[0] / self.px_range)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.transpose((2, 0, 1))
            frame = torch.from_numpy(frame).float() / 255.0  # Scale to [0, 1]
            frame = normalize(frame)
            frame = frame.to(self.device)
            frame = frame.unsqueeze(0)
        
        elif self.model_type == "resnet18":
            frame = PIL.Image.fromarray(frame)
            frame = transforms.functional.to_tensor(frame).to(self.device).half()
            # frame = frame.half()  # Convert to half precision right after to the device
            frame = frame.div_(self.px_range)  # Divide by px_range to normalize
            # frame = frame.sub_(self.tensor_mean).div_(self.tensor_stdev)
            frame = frame.unsqueeze(0)
            frame = frame.sub(self.tensor_mean).div(self.tensor_stdev)  # Normalize

        else:
            raise NotImplementedError(f"Model {self.model_type} not implemented")
        
        return frame

    def run(frame):
        raise NotImplementedError
    

class CollisionAvoidance(ModelBase):
    def __init__(self, model_path, model_type, pixel_range=255.0, RGB_mean=[0.485, 0.456, 0.406], RGB_stdev=[0.229, 0.224, 0.225], device=None):
        super().__init__(model_path, model_type, pixel_range, RGB_mean, RGB_stdev, device)
        self.threshold = 0.7 

    def run(self, frame):
        frame = self.preprocess(frame)
        frame = self.model(frame)
        frame = F.softmax(frame, dim=1)
        prob_blocked = float(frame.flatten()[0])
        logger.debug(f"Prob blocked: {round(prob_blocked, 2)}")
        
        if prob_blocked < self.threshold:
            return "forward"
        else:
            return "stop"
        
    def is_blocked(self, frame):
        action = self.run(frame)
        return True if action == "stop" else False


class PathFinder(ModelBase):
    def __init__(self, model_path, model_type, pixel_range=255.0, RGB_mean=[0.485, 0.456, 0.406], RGB_stdev=[0.229, 0.224, 0.225], device=None):
        super().__init__(model_path, model_type, pixel_range, RGB_mean, RGB_stdev, device)

    def run(self, frame):
        output = self.preprocess(frame)
        output = self.model(output)
        output = output.detach()
        output = output.float()
        output = output.cpu()
        output = output.numpy()
        output = output.flatten()

        x = output[0]
        y = (0.5 - output[1]) / 2.0
        angle = np.arctan2(x, y)
        logger.debug(f"Angle: {round(angle, 3)}")

        return angle