"""

Using a pretrained ResNet152 to classify human activities.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt


# data visualization
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    plt.show()

def trainClassifier(train_data_loader, valid_data_loader, valid_loss_min):

    for epoch in range(n_epochs):

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_data_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        model.eval()
        for data, target in valid_data_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)


        train_loss = train_loss/len(train_data_loader.dataset)
        valid_loss = valid_loss/len(valid_data_loader.dataset)

        print('[info] Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('[info] Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_activity_classification.pt')
            valid_loss_min = valid_loss

def testClassifier(test_data_loader):

    model.load_state_dict(torch.load('model_activity_classification.pt'))

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    # iterate over test data
    for data, target in test_data_loader:
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        #correct = np.squeeze(correct_tensor.numpy())
        correct = np.squeeze(correct_tensor.cpu().numpy())

        for i in range(5):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_data_loader.dataset)
    print('[info] Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            print('[info] Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('[info] Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\n[info] Test Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    # obtain one batch of test images
    dataiter = iter(test_data_loader)
    images, labels = dataiter.next()
    images.numpy()

    # move model inputs to cuda, if GPU available
    images = images.cuda()

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())

    # plot the images in the batch, along with predicted and true labels
    # fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(10):
        # ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.title('Predicted: ' + classes[preds[idx]] + '/' + 'True Label:' + classes[labels[idx]])
        imshow(images.cpu()[idx])
        # ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
        #              color=("green" if preds[idx]==labels[idx].item() else "red"))


if __name__ == '__main__':


    n_epochs = 100
    batch_size = 10
    learning_rate = 0.001

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2
    valid_loss_min = np.Inf

    train_data_path = './data/train'
    test_data_path = './data/test'
    transform_img = transforms.Compose([
                    transforms.Resize(100),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path,
                                                    transform=transform_img)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path,
                                                    transform=transform_img)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_data_loader = data.DataLoader(train_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=1)
    valid_data_loader = data.DataLoader(train_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=1)
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size,
                                        shuffle=True, num_workers=1)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[info] device in use: ', device)
    model = models.resnet152(pretrained=True)
    # print(model)

    print('[info] labels found: ',train_data.class_to_idx)
    classes = list(train_data.class_to_idx)
    print('[info] classes found: ', classes)

    '''
    # obtain one batch of training images
    dataiter = iter(train_data_loader)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display

    print('labels: ', labels)
    # plot the images in the batch, along with the corresponding labels
    # fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(20):
        print('idx: ', idx)
        print(classes[labels[idx]])
    #     img = images[idx]
    #     print('img shape: ', img.shape)
    #     plt.imshow(img[0,:,:])
    #     plt.title(classes[labels[idx]])
    #     # ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.title(classes[labels[idx]])
        imshow(images[idx])


    #     # ax.set_title(classes[labels[idx]])
    '''

    for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(512, 10),
                                         nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)


    # print('[info] training initiated..')
    # trainClassifier(train_data_loader, valid_data_loader, valid_loss_min)
    print('[info] testing initiated..')
    testClassifier(test_data_loader)
