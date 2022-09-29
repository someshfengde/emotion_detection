import streamlit as st 
import torch
from PIL import Image
import numpy as np
import torchvision.models as models
import torchvision

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_model(path):
    model=models.efficientnet_b0(pretrained=False)
    model.classifier[1]=torch.nn.Linear(in_features=1280,out_features=len(class_names))
    model.load_state_dict(torch.load("./effnet_model.pth"))
    model.eval()
    return model


def preprocess_buffer(buffer):
    image = Image.open(buffer).convert('RGB')
    image_array = np.array(image)
    resized_image = torchvision.transforms.ToTensor(image)
    return resized_image,image


def predict_and_view(buffer, model_path):
    model = load_model(model_path)
    resized_image,buffered_image = preprocess_buffer(buffer)
    prediction = model(resized_image.reshape(1,3,48,48))
    prediction_name = class_names[prediction.argmax(axis = 1).numpy()[0]]
    prediction_accuracy = prediction.max()
    return buffered_image, prediction_name, prediction_accuracy