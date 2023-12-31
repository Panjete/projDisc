## Trying to retrain the last layer(s) of the inception model to 

import torch
import torch.optim as optim
import torchvision
from torch import nn 
import os
from PIL import Image
from math import ceil

'''
Classifier Object has the following methods
learn(image_dir) -> to learn on a given database
save() -> to save state to "classifier.pth"
load() -> to load state from "classifier.pth"
forward(image_tensor) -> to yield output 18-length attributes tensor
preprocess_image(image_path) -> to read file and construct tensor




'''

img_location = "/Users/gsp/Downloads/images"

shape_labels = "/Users/gsp/Downloads/labels/shape/shape_anno_all.txt"
fabric_texture_labels = "/Users/gsp/Downloads/labels/texture/fabric_ann.txt"
pattern_texture_labels = "/Users/gsp/Downloads/labels/texture/pattern_ann.txt"

model_path = "/Users/gsp/Desktop/SemVII/COL764/projbackup/models/colored_classifier.pth"

## Returns mapping from ImageName -> Features List
def read_data(filename):
    labels_out = {}
    with open(filename, 'r') as rf:
        lines = rf.readlines()
    for line in lines:
        words = line.split()
        img_name = words[0]
        shape_vector = []
        for word in words[1:]:
            if word != "NA":
                shape_vector.append(int(word))
            else:
                shape_vector.append(-1)
        labels_out[img_name] = shape_vector
    return labels_out

# shape_labels_dict = read_data(shape_labels)
# fabric_texture_labels_dict = read_data(fabric_texture_labels)
# pattern_texture_labels_dict = read_data(pattern_texture_labels)



class Classifier(nn.Module):
    def __init__(self, shape_labels_file,  fabric_texture_file, pattern_file):
        super(Classifier, self).__init__()
        self.num_classes = 18
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="DEFAULT")
        self.last_layer = torch.nn.Linear(self.inception.fc.in_features, self.num_classes) 
        self.inception.fc = nn.Identity()
        self.inception.eval()
        
        for param in self.inception.parameters():
            param.requires_grad = False
        for param in self.last_layer.parameters():
            param.requires_grad = True

        self.shape_labels_dict = read_data(shape_labels_file)
        self.fabric_texture_labels_dict = read_data(fabric_texture_file)
        self.pattern_texture_labels_dict = read_data(pattern_file)


        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, image):
        #print("Forward recieved input shape =", image.shape)
        x = self.inception(image)
        out = self.last_layer(x).view(-1)
        return out
    
    ## Takes in full path!
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
    
    def save(self):
        torch.save(self.state_dict(), model_path)
        print("saving trained model as", model_path)# ../models/classifier.pth !")
        return
    
    def load(self):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
        else:
            print("Model does not exist, training it first")
            self.learn(img_location)
            self.save()
            self.load()

        print("loaded presaved model from ", model_path) #../models/classifier.pth !")
        return
    
    def learn(self, image_dir):
        file_list = os.listdir(image_dir)
        file_list_with_path = [os.path.join(image_dir, file) for file in file_list]
        j = 0
        for i in range(len(file_list)):
        #for i in range(500):
            img_file_name = file_list[i]
            img_file_path = file_list_with_path[i]

            if img_file_name in self.shape_labels_dict.keys() and img_file_name in self.fabric_texture_labels_dict.keys() and img_file_name in self.pattern_texture_labels_dict.keys():
                vector_img = self.shape_labels_dict[img_file_name] + self.fabric_texture_labels_dict[img_file_name] + self.pattern_texture_labels_dict[img_file_name]
                vector_tensor = torch.tensor(vector_img, dtype = torch.float).to(self.device) ## The 23 length vector

                image_processed = self.preprocess_image(img_file_path).to(self.device)
                model_output = self.forward(image_processed)

                #print("Model's output has dims = ", model_output.shape)
                #print("Tensor to train to has dims = ", vector_tensor.shape)
                self.optimizer.zero_grad()
                loss = self.criterion(model_output, vector_tensor)
                loss.backward()
                self.optimizer.step()
            else:
                j += 1
            if i %100==0:
                print("Trained on images = ", i)

        print("Total unbalanced keys = ", j)
        return
    

def returnOneHot(ourClassifier : Classifier, image_path):
    img_tensor = ourClassifier.preprocess_image(image_path)
    inferred_features = torch.Tensor.tolist(ourClassifier.forward(img_tensor))
    inferres_ints = [round(x) for x in inferred_features]
    features_length_dataset = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3, 8, 8, 8, 8, 8, 8]
    for i in range(len(inferres_ints)): ## Prune
        if inferres_ints[i] < 0:
            inferres_ints[i] = 0
        elif inferres_ints[i] >= features_length_dataset[i]:
            inferres_ints[i] = features_length_dataset[i] -1
        
    one_hot = [0 for _ in range(98)]
                            #  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11, 
    

    cur_ind = 0
    for i in range(18):
        ## We're talking about feature ith
        #print("i = ", i, "should have feature starting at = ", cur_ind, " and now, we have entry at =", inferres_ints[i])
        
        one_hot[cur_ind + inferres_ints[i]] = 1 ## If in valid range, turn this indicator on
            #print("writing 1 to position : ", cur_ind + inferres_ints[i])
        
        cur_ind += features_length_dataset[i] ## Increment pointer by len(feature)

    return one_hot

def returnTextWords(ourClassifier : Classifier, image_path):
    img_tensor = ourClassifier.preprocess_image(image_path)
    inferred_features = torch.Tensor.tolist(ourClassifier.forward(img_tensor))
    inferres_ints = [round(x) for x in inferred_features]
    features_length_dataset = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3, 8, 8, 8, 8, 8, 8]
    for i in range(len(inferres_ints)): ## Prune
        if inferres_ints[i] < 0:
            inferres_ints[i] = 0
        elif inferres_ints[i] >= features_length_dataset[i]:
            inferres_ints[i] = features_length_dataset[i] -1
    #print("Inferred Feautures = ", inferres_ints)
    words = []
                            #  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
    dictionary_features = {
        0 : {0:"sleeveless", 1 : "short-sleeve", 2 : "medium-sleeve", 3 : "long-sleeve"},
        1 : {0: "three-point lower cloth length", 1: "medium short lower cloth length", 2: "three-quarter lower cloth length", 3: "long leggings lower cloth length"},
        2 : {1 : "socks" , 2 : "leggings"}, ## Socks
        3 : {1: "hat"}, ## Hat
        4 : {1 : "eyeglasses", 2 : "sunglasses"}, ## Glasses
        5 : {1 : "neckwear"}, ## Neackwear
        6 : {1 : "wrist wearing"}, ## Wrist wearing
        7 : {1 : "ring"}, ## ring
        8 : {1 : "belt" , 2 : "clothing on waist", 3: "hidden waist"}, ## Waist accesories
        9 : {0 : "V shape neckline" , 1 : "square neckline", 2 : "round neckline", 3 : "standing neckline", 4 : "lapel neckline", 5: "suspenders neckline"}, ## Neckline
        10 : {0 : "cardigan"}, ## Cardigan?
        11 : {0 : "navel not covered" , 1: "navel covered"}, ## Navel Covered?
        12 : {0 : "denim upper fabric", 1: "cotton upper fabric", 2: "leather upper fabric", 3: "furry upper fabric", 4: "knitted upper fabric", 5: "chiffon upper fabric"}, ## Upper Fabric Annotations
        13 : {0 : "denim lower fabric", 1: "cotton lower fabric", 2: "leather lower fabric", 3: "furry lower fabric", 4: "knitted lower fabric", 5: "chiffon lower fabric"}, ## Lower Fabric Annotations
        14 : {0 : "denim outer fabric", 1: "cotton outer fabric", 2: "leather outer fabric", 3: "furry outer fabric", 4: "knitted outer fabric", 5: "chiffon upper fabric"}, ## Outer Fabric Annotations

        15: {0 : "floral upper color", 1: "graphic upper color", 2: "striped upper color", 3: "pure upper color", 4: "lattice upper color", 6: "block upper color"}, ## Upper Color
        16: {0 : "floral lower color", 1: "graphic lower color", 2: "striped lower color", 3: "pure lower color", 4: "lattice lower color", 6: "block lower color"}, ## Lower Color
        17: {0 : "floral outer color", 1: "graphic outer color", 2: "striped outer color", 3: "pure outer color", 4: "lattice outer color", 6: "block outer color"}, ## Outer Color
        
    }
    for i in range(18):
        ## We're talking about feature ith
        if i not in [2, 6, 7]: ##Not counting socks, wrist and ring
            if inferres_ints[i] in dictionary_features[i].keys(): # If encounters valid textual feature
                words += dictionary_features[i][inferres_ints[i]].split()

    return words


def a():
    ourClassifier = Classifier(shape_labels_file=shape_labels, fabric_texture_file=fabric_texture_labels, pattern_file=pattern_texture_labels)
    ourClassifier.load()
    sample = "MEN-Denim-id_00000080-01_7_additional.jpg"
    img1 = "/Users/gsp/Downloads/images/MEN-Denim-id_00000080-01_7_additional.jpg"
    img1_t = ourClassifier.preprocess_image(img1)
    print("FORWARD +++++++",ourClassifier.forward(img1_t))
    print("labels shape, fabric, pattern",ourClassifier.shape_labels_dict[sample], ourClassifier.fabric_texture_labels_dict[sample], ourClassifier.pattern_texture_labels_dict[sample])
    one_hot = returnOneHot(ourClassifier, img1)
    text_words = returnTextWords(ourClassifier, img1)
    print("one_hot encoding = ", one_hot)
    print("text words =", text_words)
    return

#a()

# print("labels shape, fabric, pattern",shape_labels_dict[sample], fabric_texture_labels_dict[sample], pattern_texture_labels_dict[sample])
# img1 = "/Users/gsp/Downloads/images/MEN-Sweatshirts_Hoodies-id_00000146-02_1_front.jpg"
# one_hot = returnOneHot(img1)
# text_words = returnTextWords(img1)
# print("one_hot encoding = ", one_hot)
# print("text words =", text_words)
# tensor1 = ourClassifier.preprocess_image(img1)
# print("Image 1 tensor shape= ", tensor1.shape)
# print("Image 1 output = ", ourClassifier.forward(tensor1))
# print("Now Training !\n\n")

# ourClassifier.learn(img_location)
# ourClassifier.save()
# img2 = "/Users/gsp/Downloads/images/MEN-Denim-id_00000080-01_7_additional.jpg"
# tensor2 = ourClassifier.preprocess_image(img2)
# print("Image 1 after training yields : ", ourClassifier.forward(tensor2))
        
#[      5,       3,       0,       0,       0,       0,       0,       0,       3,       2,       1,       1       1,       1,       7        3,       4,       7]
#[-0.0353,  0.0415, -0.0474,  0.0147,  0.0089, -0.1455, -0.3389, -0.2938, -0.2596, -0.2454, -0.2570, -0.4066, 0.3069, -0.0288, -0.0545, -0.0691, -0.2012,  0.2981]
#[ 1.0563,  2.8451,  0.5471,  0.0428, -0.3958, -0.1766, -0.2021,  0.5303, 1.9068,  2.6096,  1.0149,  1.1397,  1.3911,  0.7264,  7.1941,  4.4880, 3.3402,  7.3134]
