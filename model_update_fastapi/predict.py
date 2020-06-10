# reference : https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

from torchvision import models, transforms
import torch
from PIL import Image

def predicted_results(img, pymodel):
    img = Image.open(img)
    
    transform = transforms.Compose([           #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
    )])
    
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        
    pymodel.eval()
    
    out = pymodel(batch_t)
    # Forth, print the top 5 classes predicted by the model
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    
    # Third, carry out model inference

    return {
        "model_name" : pymodel.__class__.__name__,
        "predicted_results" : [{classes[idx] : percentage[idx].item()} for idx in indices[0][:5]]
    }


        
    
    
    
    