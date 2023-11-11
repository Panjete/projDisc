## Trying to output the second last layer as visual features

import torch
import torch.optim as optim
import torchvision
from torch import nn 
import os
from PIL import Image

'''
Visualiser has the following methods:

forward(image) -> to get visual features from image tensor
load() -> to load from "visualiser.pth"
save() -> to save to "visualiser.pth"
preprocess_image -> to convert image to suitably shaped tensor

'''

class Visualiser(nn.Module):
    def __init__(self):
        super(Visualiser, self).__init__()
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="DEFAULT")
        self.inception.fc = nn.Identity()
        self.inception.eval()
        
        for param in self.inception.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, image):
        out = self.inception(image).view(-1)
        return out
    
    def save(self):
        torch.save(self.state_dict(), "visualiser.pth")
        print("saving trained model as visualiser.pth !")
        return
    def load(self):
        self.load_state_dict(torch.load("visualiser.pth"))
        print("loaded presaved model from visualiser.pth !")
        return
    
    def preprocess_image(self, image_filename):
        input_image = Image.open(image_filename)
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(299),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch



sample = "MEN-Denim-id_00000080-01_7_additional.jpg"
img1 = "/Users/gsp/Downloads/images/MEN-Denim-id_00000080-01_7_additional.jpg"


## Gives a list of 2048-length visual features from image_path 
def returnVisualFeatures(ourVisualiser : Visualiser, image_path):
    tensor1 = ourVisualiser.preprocess_image(image_path)
    image_results = ourVisualiser.forward(tensor1)
    feature_list = torch.Tensor.tolist(image_results)
    return feature_list
    

def b():
    ourVisualiser = Visualiser()
    #ourVisualiser.save()
    ourVisualiser.load()
    visual_features = returnVisualFeatures(ourVisualiser, img1)
    print(len(visual_features))


#b()









