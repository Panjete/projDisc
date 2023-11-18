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
save() -> to save state to "models/classifier.pth"
load() -> to load state from "models/classifier.pth"
forward(image_tensor) -> to yield output 98-length attributes tensor
preprocess_image(image_path) -> to read file and construct tensor

'''

img_location = "/Users/gsp/Downloads/images"
shape_labels = "/Users/gsp/Downloads/labels/shape/shape_anno_all.txt"
fabric_texture_labels = "/Users/gsp/Downloads/labels/texture/fabric_ann.txt"
pattern_texture_labels = "/Users/gsp/Downloads/labels/texture/pattern_ann.txt"

model_path = "/Users/gsp/Desktop/SemVII/COL764/projbackup/models/classifier_vgg.pth"

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

class Classifier_vgg(nn.Module):
    def __init__(self, shape_labels_file,  fabric_texture_file, pattern_file):
        super(Classifier_vgg, self).__init__()
        self.num_classes = 98 ## Training rather over one-hot
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', weights = "DEFAULT")
        
        in_features = self.vgg.classifier[-1].in_features
        self.last_layer = torch.nn.Linear(in_features, self.num_classes) 
        self.sg = torch.nn.Sigmoid()
        self.vgg.classifier[-1] = nn.Identity()
        self.vgg.eval()
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        for param in self.last_layer.parameters():
            param.requires_grad = True

        self.shape_labels_dict = read_data(shape_labels_file)
        self.fabric_texture_labels_dict = read_data(fabric_texture_file)
        self.pattern_texture_labels_dict = read_data(pattern_file)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.BCELoss(reduction= 'sum')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, image):
        #print("Forward recieved input shape =", image.shape)
        x = self.vgg(image)    
        out = self.last_layer(x).view(-1)
        out = self.sg(out)
        return out
    
    ## Takes in full path!
    def preprocess_image(self, image_filename):
        input_image = Image.open(image_filename)
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch
    
    def save(self):
        torch.save(self.state_dict(),model_path)
        print("saving trained model as " , model_path)
        return
    
    def load(self):
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print("loaded presaved model from " , model_path)
        else:
            print("Model does not exist, training it first")
            self.learn(img_location)
            self.save()
            self.load()
            print("saving!")
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
                one_hot_features = features_to_one_hot(vector_img)
                one_hot_tensor = torch.tensor(one_hot_features, dtype = torch.float).to(self.device) ## The 98 length vector


                image_processed = self.preprocess_image(img_file_path).to(self.device)
                model_output = self.forward(image_processed)

                #print("Model's output has dims = ", model_output.shape)
                #print("Tensor to train to has dims = ", vector_tensor.shape)
                self.optimizer.zero_grad()
                loss = self.criterion(model_output, one_hot_tensor)
                loss.backward()
                self.optimizer.step()
            else:
                j += 1
            if i %100==0:
                print("Trained on images = ", i)

        print("Total unbalanced keys = ", j)
        return
    
def features_to_one_hot(image_features):
    features_length_dataset = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3, 8, 8, 8, 8, 8, 8]
    for i in range(len(image_features)): ## Clip
        if image_features[i] < 0:
            image_features[i] = 0
        elif image_features[i] >= features_length_dataset[i]:
            image_features[i] = features_length_dataset[i] -1
        
    one_hot = [0 for _ in range(98)]
    cur_ind = 0
    for i in range(18):
        ## We're talking about feature ith
        #print("i = ", i, "should have feature starting at = ", cur_ind, " and now, we have entry at =", inferres_ints[i])
        one_hot[cur_ind + image_features[i]] = 1 ## If in valid range, turn this indicator on
        cur_ind += features_length_dataset[i] ## Increment pointer by len(feature)

    return one_hot

## Simply returns models output as pythonic array
def returnOneHot(ourClassifier : Classifier_vgg, image_path):
    img_tensor = ourClassifier.preprocess_image(image_path)
    inferred_features = torch.Tensor.tolist(ourClassifier.forward(img_tensor))
    inferres_ints = [round(x) for x in inferred_features]
    return inferres_ints

## Convert 98 length to 18 lenght for text words extraction
def one_hot_to_features(ourClassifier : Classifier_vgg, image_path):
    img_tensor = ourClassifier.preprocess_image(image_path)
    inferred_features = torch.Tensor.tolist(ourClassifier.forward(img_tensor)) ## length 98
    features_length_dataset = [6, 5, 4, 3, 5, 3, 3, 3, 5, 7, 3, 3, 8, 8, 8, 8, 8, 8]

    features = []
    cur_ind = 0
    for i in range(18): ## For all features
        max_el = -1
        max_at = -1
        for j in range(cur_ind, cur_ind + features_length_dataset[i]): ## Check the range spanned by this feature
            if inferred_features[j] > max_el:
                max_el = inferred_features[j]
                max_at = j

        features.append(max_at-cur_ind)
        cur_ind += features_length_dataset[i]
        
        

    return features

def returnTextWords(ourClassifier : Classifier_vgg, image_path):
    inferres_ints = one_hot_to_features(ourClassifier, image_path)
    words = []
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
    ourClassifier = Classifier_vgg(shape_labels_file=shape_labels, fabric_texture_file=fabric_texture_labels, pattern_file=pattern_texture_labels)
    
    sample = "MEN-Denim-id_00000080-01_7_additional.jpg"
    img1 = "/Users/gsp/Downloads/images/MEN-Denim-id_00000080-01_7_additional.jpg"
    img1_t = ourClassifier.preprocess_image(img1)
    #print("FORWARD +++++++",ourClassifier.forward(img1_t))
    print("labels shape, fabric, pattern",ourClassifier.shape_labels_dict[sample], ourClassifier.fabric_texture_labels_dict[sample], ourClassifier.pattern_texture_labels_dict[sample])
    one_hot = returnOneHot(ourClassifier, img1)
    features = one_hot_to_features(ourClassifier, img1)
    text_words = returnTextWords(ourClassifier, img1)
    #print("one_hot encoding = ", one_hot)
    print("text words =", text_words)
    print("features = ", features)
    ourClassifier.load()
    print("AFTER TRAINING")
    #print("FORWARD +++++++",ourClassifier.forward(img1_t))
    one_hot = returnOneHot(ourClassifier, img1)
    text_words = returnTextWords(ourClassifier, img1)
    features = one_hot_to_features(ourClassifier, img1)
    #print("one_hot encoding = ", one_hot)
    print("text words =", text_words)
    print("features = ", features)
    
    
    return

#a()