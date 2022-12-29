import cv2 as cv
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn.functional as F
import torchvision.transforms
from torchvision import transforms, datasets
from model import *

if __name__ == '__main__':
    model = TridentNetwork()
    model.load_state_dict(torch.load('./Storage_model/TridentNetwork'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model.eval()

    image_datasets = datasets.ImageFolder('./data/wild_fire_test', transform=transform)
    test_loader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=True, num_workers=0)
    fp = 0
    fn = 0
    p_samples = image_datasets.targets.count(image_datasets.class_to_idx['nofire'])
    n_samples = image_datasets.targets.count(image_datasets.class_to_idx['fire'])

    for _, (img, label) in enumerate(test_loader):
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        diff = label - preds
        print(diff)
        if diff.item() < 0:
            fp += 1
        elif diff.item() > 0:
            fn += 1
        else:
            pass
        break
    tp = p_samples - fn
    tn = n_samples - fp

    # Sensitive
    TPR = tp / p_samples
    # Specificity
    TNR = tn / n_samples
    # Precision
    Precision = tp / (tp + fp)
