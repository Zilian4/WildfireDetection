# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torchvision
import torch
import torch.nn as nn
import sys
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import time
import os
import copy
from model import *
import matplotlib.pyplot as plt


def plot_loss_epoch(epochs, line1, line2):
    plt.rc('axes', facecolor='white')
    plt.rc('figure', figsize=(6, 4))
    plt.rc('axes', grid=False)
    plt.plot(range(1, epochs + 1), line1, '.:r', range(1, epochs + 1), line2, ':b')
    plt.legend(['Train', "Validation"], loc='upper right')

    plt.title('Loss-Epoch')
    plt.ylabel('Loss')

    plt.xlabel('Epoch')
    plt.show()


def plot_acc_epoch(epochs, line1, line2):
    plt.rc('axes', facecolor='white')
    plt.rc('figure', figsize=(6, 4))
    plt.rc('axes', grid=False)
    plt.plot(range(1, epochs + 1), line1, '.:r', range(1, epochs + 1), line2, ':b')
    plt.legend(['Train', "Validation"], loc='upper right')

    plt.title('Accuracy-Epoch')
    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')
    plt.show()



def train_model(model, criterion, optimizer, scheduler, num_epochs=15, freeze_epoch=-1):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Save losses and accuracy of each epoch
    loss_list_train = []
    acc_list_train = []
    loss_list_val = []
    acc_list_val = []

    for epoch in range(num_epochs):

        # Freeze model
        if epoch < freeze_epoch:
            model.set_freeze(True)
        else:
            model.set_freeze(False)

        print(f'Epoch {epoch}/{num_epochs - 1},(Freeze Condition:{model.is_freeze})')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        print('\rTraining Progress:{:.2f}%'.format(100 * count / dataset_sizes['train']),
                              "-" * (10 * count // dataset_sizes['train']), end='')
                        sys.stdout.flush()
                        time.sleep(0.01)
                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                count += dataloaders['train'].batch_size

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()
                loss_list_train.append(epoch_loss)
                acc_list_train.append(epoch_acc.cpu())
            else:
                loss_list_val.append(epoch_loss)
                acc_list_val.append(epoch_acc.cpu())

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    time_elapsed = time.time() - since

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # show the linechart of loss and acc vs epochs in train set and val set
    plot_loss_epoch(num_epochs, loss_list_train, loss_list_val)
    plot_acc_epoch(num_epochs, acc_list_train, acc_list_val)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    cudnn.benchmark = True
    # plt.ion()  # interactive mode

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = './data/wild_fire_train'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = 'TridentNetwork'

    model = get_model(model_name)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20,)

    torch.save(model.state_dict(), "./Storage_model/" + str(model_name))
    model.load_state_dict(torch.load('./Storage_model/TridentNetwork'))

