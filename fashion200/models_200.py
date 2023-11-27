## For the Fashion 200K Dataset

import torch
import torchvision
from torch import nn 
import os
from PIL import Image
from math import ceil
import json

img_location = "/Users/gsp/Downloads/women"
labels_location = "/Users/gsp/Downloads/labels_200"
json_file_path = "captions_200.json"
model_path = "/Users/gsp/Desktop/SemVII/COL764/projbackup/models/visualiser_200.pth"
'''
Visualiser has the following methods:

forward(image) -> to get visual features from image tensor
load() -> to load from "visualiser_200.pth"
save() -> to save to "visualiser_200.pth"
preprocess_image -> to convert image to suitably shaped tensor

'''


## For collecting image paths recursively from the folder
def get_image_paths_relative():
    file_paths = []
    for foldername, _, filenames in os.walk(img_location):
        for filename in filenames:
            if filename[0]!= ".": 
                file_path = os.path.join(foldername, filename)
                file_paths.append(file_path[21:])
    return file_paths

## Wrt PC (contains /Users/gsp/ etc)
def get_image_paths_absolute():
    file_paths = []
    for foldername, _, filenames in os.walk(img_location):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            file_paths.append(file_path)
    return file_paths

## Returns mapping from ImageName -> Features List
## Used for reading "label" files
def read_data(filename):
    labels_out = {}
    with open(filename, 'r') as rf:
        lines = rf.readlines()
    for line in lines:
        words = line.split()
        img_name = words[0]
        shape_vector = []
        for word in words[2:]:
            shape_vector.append(word)
        labels_out[img_name] = shape_vector
    print("read data for ", filename)
    return labels_out


def collect_data():
    filenames = [labels_location + x for x in ["/dress.txt", "/top.txt", "/jacket.txt", "/pants.txt", "/skirt.txt"]]
    img_paths = get_image_paths_relative()

    final_captions = {}

    dicts_dress = read_data(filenames[0]) 
    dicts_top = read_data(filenames[1]) 
    dicts_jacket = read_data(filenames[2]) 
    dicts_pants = read_data(filenames[3]) 
    dicts_skirt = read_data(filenames[4]) 

    for img in img_paths:
        words = []
        if img in dicts_dress:
            words += dicts_dress[img]
        if img in dicts_top:
            words += dicts_top[img] 
        if img in dicts_jacket:
            words += dicts_jacket[img] 
        if img in dicts_pants:
            words += dicts_pants[img] 
        if img in dicts_skirt:
            words += dicts_skirt[img] 

        final_captions[img] = " ".join(words)

    with open(json_file_path, 'w') as json_file:
        json.dump(final_captions, json_file, indent=4)

    print("All captions written!")
    return 




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
        torch.save(self.state_dict(), model_path)
        print("saving trained model as " , model_path)
        return
    def load(self):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load("models/visualiser_200.pth"))
        else:
            print("Model does not exist, training it first")
            self.save()
            self.load()

        print("loaded presaved model from " , model_path)
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
    
    def preprocess_PIL(self, image_PIL):
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(299),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image_PIL)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch



img1 = "/Users/gsp/Downloads/images/MEN-Denim-id_00000080-01_7_additional.jpg"


## Gives a list of 2048-length visual features from image_path 
def returnVisualFeatures(ourVisualiser : Visualiser, image_path):
    tensor1 = ourVisualiser.preprocess_image(image_path)
    image_results = ourVisualiser.forward(tensor1)
    feature_list = torch.Tensor.tolist(image_results)
    return feature_list

## Gives a list of 2048-length visual features from image_PIL 
def returnVisualfromPIL(ourVisualiser : Visualiser, image_PIL):
    tensor1 = ourVisualiser.preprocess_PIL(image_PIL)
    image_results = ourVisualiser.forward(tensor1)
    feature_list = torch.Tensor.tolist(image_results)
    return feature_list
    

def b():
    ourVisualiser = Visualiser()
    #ourVisualiser.save()
    ourVisualiser.load()
    visual_features = returnVisualFeatures(ourVisualiser, img1)
    print(len(visual_features))

#collect_data()

#b()









