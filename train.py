import copy
import torch
import itertools
import numpy as np
import torch.nn as nn
from torch import optim
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch
from torch.autograd import Variable
import copy
import math

import dataset, models

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion() #interactive mode
batch_size = 8

def setup_env(ds_name, metric_name, net_id, train_mode, is_hec, ds_perc,isFnT):

    if ds_name == "car_accidents":
        num_classes = 2
    elif ds_name =="cifar":
        num_classes = 10
    elif ds_name == "birds":
        num_classes = 525
    else:
       print("Error in num of classes!!")

    print("The mode is fintetuned?{}".format(isFnT))
    # create our ML model
    if net_id == 0:  # AlexNet 3:#SwinTransformer 4:#VGG 5:#REGNet
        model, loss_fn, img_size = models.AlexNet(ds_name, num_classes, isFnT)
    elif net_id == 1:#EfficientNet
        model, loss_fn, img_size = models.EfficientNet_b0(ds_name, num_classes, isFnT)
    elif net_id == 2:#ViT
        model, loss_fn, img_size = models.ViT(ds_name, num_classes, isFnT)
    elif net_id == 3:#SwinTransformer
        model, loss_fn, img_size = models.SwinTransformer(ds_name, num_classes, isFnT)
    elif net_id == 4:#VGG
        model, loss_fn, img_size = models.VGG(ds_name, num_classes, isFnT)
    elif net_id == 5:#REGNet
        model, loss_fn, img_size = models.REGNet(ds_name, num_classes, isFnT)
    elif net_id == 30:
        model, loss_fn, img_size = models.ViT_scratch(ds_name, num_classes)
    else:
        print("net_id should be in [0, 3], Error!")

    # let's do some data augmentation to improve our synthetic dataset
    trans_train = transforms.Compose([
      transforms.CenterCrop([img_size,img_size]),
      transforms.ToTensor()])

    trans_val_test = transforms.Compose([
      transforms.CenterCrop([img_size, img_size]),
      # transforms.CenterCrop([300, 300]),
      transforms.ToTensor(),])


# all datasets have n train sampels expcept (Syn* + Rel) where we have 0.5Rel+0.25Synt
    if train_mode == -1: #20 (Synth nonphoto)
        train_loaders, val_loader, test_loader = dataset.train_synthetic_only_nonphoto(ds_name,batch_size, trans_train, trans_val_test,is_hec,ds_perc)
    if train_mode == 0:  # 0 (Syn), 1 (Rel), 2 (Syn+Rel), 3(Syn+ + Rel), 4(Syn + Rel+), 5(Syn* + Rel), 20 (Synth nonphoto)
        train_loaders, val_loader, test_loader = dataset.train_synthetic_only(ds_name,batch_size, trans_train, trans_val_test,is_hec,ds_perc)
    elif train_mode == 1:
        train_loaders, val_loader, test_loader = dataset.train_real_only(ds_name,batch_size, trans_train, trans_val_test,is_hec,ds_perc)
    elif train_mode == 2:#equal to synth only;equal to real only
        train_loaders, val_loader, test_loader = dataset.train_synthetic_real(ds_name,batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 3: # 3(Syn+ + Rel) 0.75+0.25
        train_loaders, val_loader, test_loader = dataset.train_syntheticPLS_real(ds_name,batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 4: # 4(Syn + Rel+)
        train_loaders, val_loader, test_loader = dataset.train_synthetic_realPLS(ds_name,batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 5: # 5(Syn*0.25 + Rel0.5)
        train_loaders, val_loader, test_loader = dataset.train_syntheticAst_real(ds_name,metric_name,batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 6: # 6(Syn*0.5 + Rel0.5)
        train_loaders, val_loader, test_loader = dataset.train_syntheticAst_real2(ds_name,metric_name,batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 7: # 7(SynBad* + Rel)
        train_loaders, val_loader, test_loader = dataset.train_syntheticAstBad_real(ds_name,batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 8: # 8(SynArt* + Rel)
        train_loaders, val_loader, test_loader = dataset.train_syntheticAst_real_as(ds_name,batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 10:
        train_loaders, val_loader, test_loader = dataset.train_synthetic_ourApproach(ds_name, batch_size, trans_train, trans_val_test,is_hec)
    elif train_mode == 21: #21 (Synth nonphoto + real)
        train_loaders, val_loader, test_loader = dataset.train_synthetic_real_nonphoto(ds_name,batch_size, trans_train, trans_val_test,is_hec)

    # define the optimizer
    # optimizer = Adam(model.parameters(),lr=0.1,weight_decay=1e-5)# for all others except birds Alex S*+R
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    # for  birds Alex S*+R
    ##scheduler = ExponentialLR(optimizer, gamma=0.95)

    if ds_name == "car_accidents" and net_id == 0:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    elif ds_name == "car_accidents" and net_id == 1:
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    else:
    # Before 2024 May
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)


    # define our execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")

    return model, device, optimizer, loss_fn, train_loaders, val_loader, test_loader, scheduler
# save the model
def save_model(model,model_path, epoch):
    # print(model_path)
    # torch.save(model.state_dict(), model_path)
    print("Save nothing!")


# # test the model with the test dataset and print the accuracy for the test images
# def valid_accuracy_multiClss(model,val_loader,device):
#     model.eval()
#
#     total = 0.0
#     correct = 0
#     with torch.no_grad():
#         for (images, labels) in val_loader:
#
#             images = Variable(images.to(device))
#             labels = Variable(labels.to(device))
#
#             # run the model on the test set to predict labels
#             outputs = model(images)
#
#             ret, predicted = torch.max(outputs.data, 1)
#             correct += predicted.eq(labels.data).sum().item()
#             total += labels.size(0)
#             # print(correct)
#             # print(total)
#
#
#     # compute the accuracy over all test images
#     accuracy = 100 * correct // total
#     return accuracy

def valid_accuracy(model,val_loader,device):
    model.eval()

    total = 0.0
    correct = 0
    # m = nn.Sigmoid()
    with torch.no_grad():
        for (images,labels) in val_loader:

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # run the model on the test set to predict labels
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = 100 * correct // total
    return accuracy

# Approach 1 using just alternative training
# training function.
# we simply have to loop over our data iterator and feed the inputs to the network and optimize.
# def train_binary(num_epochs, ds_name, metric_name, net_id,train_mode,is_hec,model_path2save,ds_perc, isFnT):
#
#     model, device, optimizer, loss_fn, train_loaders, val_loader, test_loader, scheduler = setup_env(ds_name, metric_name,
#                                                                                                     net_id, train_mode,
#                                                                                                     is_hec, ds_perc,isFnT)
#
#     best_accuracy = -1.0
#     best_model = []
#     losses = []
#     accuracy_val = []
#     # send our model to CPU or GPU
#     model.to(device)
#     change_score = False
#     # m = nn.Sigmoid()
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
#
#         # if epoch%2==0:
#         #     train_loader = train_loaders[0]
#         # else:
#         #     train_loader = train_loaders[1]
#
#         # if change_score:
#         #     train_loader = train_loaders[0]
#         # else:
#         #     train_loader = train_loaders[1]
#
#         for i, (images, labels) in enumerate(train_loader):
#
#             images = Variable(images.to(device))
#             labels = Variable(labels.to(device))
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # predict classes using images from the training set
#             outputs = model(images)
#
#             # compute the loss based on model output and real labels
#             loss = loss_fn(outputs, labels)
#             # back-propagate the loss
#             loss.backward()
#             # print(loss.item())
#             # adjust parameters based on the calculated gradients
#             optimizer.step()
#
#             running_loss += loss.item()  # extract the loss value
#
#         # save training losses to plot them at the end of the training
#         losses.append(running_loss)
#
#         # print loss at the end of each epoch
#         print('-------------------------------------------')
#         print('Epoch %d, train loss: %.3f' % (epoch + 1, running_loss ))
#
#         # compute and print the average validation accuracy
#         accuracy = valid_accuracy_binary(model,val_loader,device)
#
#         # save validation accuracy to plot them at the end of the training
#         accuracy_val.append(accuracy)
#
#         print('The validation accuracy on the valid set (real) is %d %%' % accuracy)
#
#         # we want to save the model if the accuracy is the best
#         if accuracy > best_accuracy:
#             print('Best validation accuracy on the valid set (real) is %d %%' % accuracy)
#             # save_model(model,model_path2save, epoch)
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(model)
#
#     return model,best_model, best_accuracy, test_loader, losses, accuracy_val

# Approach 2 using simple max algorithm
# def train_binary(num_epochs, ds_name, metric_name, net_id, train_mode, is_hec, model_path2save, ds_perc, isFnT,
#                  patience=3):
#     model, device, optimizer, loss_fn, train_loaders, val_loader, test_loader, scheduler = setup_env(ds_name,
#                                                                                                      metric_name,
#                                                                                                      net_id, train_mode,
#                                                                                                      is_hec, ds_perc,
#                                                                                                      isFnT)
#
#     best_accuracy = -1.0
#     best_model = []
#     losses = []
#     accuracy_val = []
#     model.to(device)
#
#     change_score = 0
#     epochs_no_improve = 0  # Counter for epochs with no improvement
#     num_loaders = len(train_loaders)
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         train_loader = train_loaders[change_score]
#
#         for i, (images, labels) in enumerate(train_loader):
#             images = Variable(images.to(device))
#             labels = Variable(labels.to(device))
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = loss_fn(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         losses.append(running_loss)
#         print('-------------------------------------------')
#         print(f'Epoch {epoch + 1}, train loss: {running_loss:.3f}')
#
#         accuracy = valid_accuracy_binary(model, val_loader, device)
#         accuracy_val.append(accuracy)
#         print(f'The validation accuracy on the valid set (real) is {accuracy} %')
#
#         if accuracy > best_accuracy:
#             print(f'Best validation accuracy on the valid set (real) is {accuracy} %')
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(model)
#             epochs_no_improve = 0  # Reset counter if accuracy improves
#         else:
#             epochs_no_improve += 1  # Increment counter if no improvement
#
#         if epochs_no_improve >= patience:
#             # Evaluate all loaders
#             best_loader_accuracy = -1
#             best_loader_index = change_score
#
#             for loader_index in range(num_loaders):
#                 loader = train_loaders[loader_index]
#                 temp_accuracy = valid_accuracy_binary(model, loader, device)
#                 if temp_accuracy > best_loader_accuracy:
#                     best_loader_accuracy = temp_accuracy
#                     best_loader_index = loader_index
#
#             # Switch to the best loader found
#             change_score = best_loader_index
#             epochs_no_improve = 0  # Reset counter after switching loader
#             print(
#                 f'No improvement for {patience} epochs. Switching to data loader {change_score} with validation accuracy {best_loader_accuracy} %.')
#
#     return model, best_model, best_accuracy, test_loader, losses, accuracy_val


# Approach 3 using MultiArm Bandi
# class UCBLoaderSelector:
#     def __init__(self, num_loaders, c=2):
#         self.num_loaders = num_loaders
#         self.c = c
#         self.loader_counts = [0] * num_loaders
#         self.loader_rewards = [0.0] * num_loaders
#         self.total_counts = 0
#
#     def select_loader(self):
#         if self.total_counts < self.num_loaders:
#             # Ensure each loader is selected at least once
#             return self.total_counts
#
#         ucb_values = [
#             self.loader_rewards[i] / self.loader_counts[i] + self.c * math.sqrt(
#                 math.log(self.total_counts) / self.loader_counts[i])
#             for i in range(self.num_loaders)
#         ]
#         return ucb_values.index(max(ucb_values))
#
#     def update_rewards(self, loader_index, reward):
#         self.loader_counts[loader_index] += 1
#         self.loader_rewards[loader_index] += reward
#         self.total_counts += 1
#
# # patience = 3, c = 2
# def train_binary(num_epochs, ds_name, metric_name, net_id, train_mode, is_hec, model_path2save, ds_perc, isFnT,
#                  patience=3, ucb_c=2):
#     model, device, optimizer, loss_fn, train_loaders, val_loader, test_loader, scheduler = setup_env(ds_name,
#                                                                                                      metric_name,
#                                                                                                      net_id, train_mode,
#                                                                                                      is_hec, ds_perc,
#                                                                                                      isFnT)
#
#     best_accuracy = -1.0
#     best_model = []
#     losses = []
#     accuracy_val = []
#     model.to(device)
#
#     ucb_selector = UCBLoaderSelector(num_loaders=len(train_loaders), c=ucb_c)
#     epochs_no_improve = 0
#
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         loader_index = ucb_selector.select_loader()
#         train_loader = train_loaders[loader_index]
#         print(f"loader {loader_index} is selected!")
#
#         for i, (images, labels) in enumerate(train_loader):
#             images = Variable(images.to(device))
#             labels = Variable(labels.to(device))
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = loss_fn(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         losses.append(running_loss)
#         print('-------------------------------------------')
#         print(f'Epoch {epoch + 1}, train loss: {running_loss:.3f}')
#
#         accuracy = valid_accuracy_binary(model, val_loader, device)
#         accuracy_val.append(accuracy)
#         print(f'The validation accuracy on the valid set (real) is {accuracy} %')
#
#         ucb_selector.update_rewards(loader_index, accuracy)
#
#         if accuracy > best_accuracy:
#             print(f'Best validation accuracy on the valid set (real) is {accuracy} %')
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(model)
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#
#         if epochs_no_improve >= patience:
#             best_loader_index = ucb_selector.select_loader()
#             epochs_no_improve = 0
#             print(f'No improvement for {patience} epochs. Switching to data loader {best_loader_index}.')
#
#     return model, best_model, best_accuracy, test_loader, losses, accuracy_val

# Approach 4 Multi Arm bandit with ruobin

class UCBLoaderSelector:
    def __init__(self, num_loaders, c):
        self.num_loaders = num_loaders
        self.loader_counts = np.zeros(num_loaders)
        self.loader_rewards = np.zeros(num_loaders)
        self.total_counts = 0
        self.c = c

    def select_loader(self):
        if self.total_counts < self.num_loaders:
            return self.total_counts  # Initially select each loader once
        ucb_values = self.loader_rewards / (self.loader_counts + 1e-5) + self.c * np.sqrt(np.log(self.total_counts) / (self.loader_counts + 1e-5))
        return np.argmax(ucb_values)

    def update_rewards(self, loader_index, reward):
        self.loader_counts[loader_index] += 1
        self.loader_rewards[loader_index] += reward
        self.total_counts += 1

def train(num_epochs, ds_name, metric_name, net_id, train_mode, is_hec, model_path2save, ds_perc, isFnT, patience=3, ucb_c=2):
    model, device, optimizer, loss_fn, train_loaders, val_loader, test_loader, scheduler = setup_env(ds_name, metric_name, net_id, train_mode, is_hec, ds_perc, isFnT)

    best_accuracy = -1.0
    best_model = None
    losses = []
    accuracy_val = []
    model.to(device)

    ucb_selector = UCBLoaderSelector(num_loaders=len(train_loaders), c=ucb_c)
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        loader_index = ucb_selector.select_loader()
        train_loader = train_loaders[loader_index]
        print(f"loader {loader_index} is selected!")

        for images, labels in train_loader:
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        losses.append(running_loss)
        print('-------------------------------------------')
        print(f'Epoch {epoch + 1}, train loss: {running_loss:.3f}')

        accuracy = valid_accuracy(model, val_loader, device)
        accuracy_val.append(accuracy)
        print(f'The validation accuracy on the valid set (real) is {accuracy} %')

        ucb_selector.update_rewards(loader_index, accuracy)

        if accuracy > best_accuracy:
            print(f'Best validation accuracy on the valid set (real) is {accuracy} %')
            best_accuracy = accuracy
            best_model = copy.deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            best_loader_index = ucb_selector.select_loader()
            epochs_no_improve = 0
            print(f'No improvement for {patience} epochs. Switching to data loader {best_loader_index}.')

    return model, best_model, best_accuracy, test_loader, losses, accuracy_val

# training function.
# we simply have to loop over our data iterator and feed the inputs to the network and optimize.
# def train_multiClass(num_epochs,ds_name,metric_name, net_id,train_mode,is_hec,model_path2save,ds_perc,isFnT):
#
#     model, device, optimizer, loss_fn, train_loader, val_loader, test_loader,scheduler = setup_env(ds_name,metric_name,net_id, train_mode,is_hec,ds_perc,isFnT)
#
#     best_accuracy = -1.0
#     best_model = []
#     losses = []
#     accuracy_val = []
#     # send our model to CPU or GPU
#     model.to(device)
#
#     for epoch in range(num_epochs):  # loop over the dataset multiple times
#         running_loss = 0.0
#
#         for i, (images,labels) in enumerate(train_loader):
#
#             images = Variable(images.to(device))
#             labels = Variable(labels.to(device))
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # predict classes using images from the training set
#             outputs = model(images)
#
#             # compute the loss based on model output and real labels
#             loss = loss_fn(outputs, labels)
#
#             # back-propagate the loss
#             loss.backward()
#
#             # adjust parameters based on the calculated gradients
#             optimizer.step()
#
#             # print(loss.item())
#             running_loss += loss.item()  # extract the loss value
#
#         scheduler.step(running_loss)
#         # save training losses to plot them at the end of the training
#         losses.append(running_loss)
#
#         # print loss at the end of each epoch
#         print('-------------------------------------------')
#         print('Epoch %d, train loss: %.3f' % (epoch + 1, running_loss ))
#
#         # compute and print the average validation accuracy
#         accuracy = valid_accuracy_multiClss(model, val_loader, device)
#
#         # save validation accuracy to plot them at the end of the training
#         accuracy_val.append(accuracy)
#
#         print('The validation accuracy on the valid set (synth + real) is %d %%' % accuracy)
#
#         # we want to save the model if the accuracy is the best
#         if accuracy > best_accuracy:
#             print('Best validation accuracy on the valid set (synth + real) is %d %%' % accuracy)
#             # save_model(model,model_path2save, epoch)
#             best_accuracy = accuracy
#             best_model = copy.deepcopy(model)
#
#     return model,best_model, best_accuracy, test_loader, losses, accuracy_val