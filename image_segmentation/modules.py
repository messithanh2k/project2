import time

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.autograd import Variable


def train_model(epoch, model, data_loader, criterion, optimizer, scheduler, device):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        train_loader (DataLoader): training dataset
    """
    model.train()
    st = time.time()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        # print("shapeeeeeee: ", images.shape, labels.shape)

        outputs = model(images)
        optimizer.zero_grad()
        # print(images.shape)
        # print(labels.shape)
        # print(outputs.shape)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        et = time.time()
        loss_value = loss.item()
        total_loss += loss_value
        log_str = 'epoch {}:({}/{}) || time {:.5f} || loss {:5f} || total loss {:5f} '.format(epoch, batch_idx,
                                                                                              len(
                                                                                                  data_loader),
                                                                                              et - st,
                                                                                              loss_value, total_loss)
        print(log_str)
        st = time.time()

    scheduler.step(total_loss)
    return total_loss


def evaluate_model(model, data_loader, criterion, device, metric=False):
    """
        Calculate loss over train set
    """
    total_loss = 0
    all_acc = []
    # all_pixels  = []

    model.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            # print("shapeeeeeee: ",images.shape,labels.shape)

            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            # 
            outputs = outputs.cpu().numpy()
            predicted = np.argmax(outputs, axis=1)

            # 
            labels = labels.cpu().numpy()
            # labels      = np.argmax(labels, axis=1)

            # 
            # all_pixels.append(labels.shape[0] * labels.shape[1])

            # calculate acc
            matches = (predicted == labels).astype(np.uint8)
            all_acc.append(np.sum(matches) / (labels.shape[1] * labels.shape[2]))
    print("lennnnnnn: ", len(data_loader), len(all_acc))

    return total_loss / len(data_loader), np.mean(np.array(all_acc))