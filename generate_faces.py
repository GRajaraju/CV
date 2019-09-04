import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

image_size = 64 
batchSize = 64 


transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.ToTensor(),
                                ])

data_dir='./processed-celeba-small/'

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                    shuffle=True)

def weight_init(m):
  classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
        

# defining the generator
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                    nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                    nn.Tanh()
                    )

    def forward(self, x):
        output = self.main(x)
        return output

# creating the generator
net_gen = Generator()

# defining the discriminator
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                    )
    def forward(self, x):
        output = self.main(x)
        return output.view(-1)


# creating the discriminator
net_dis = Discriminator()

# training the dcgan
criterion = nn.BCELoss()
optimizer_dis = optim.Adam(net_dis.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_gen = optim.Adam(net_gen.parameters(), lr=0.0002, betas=(0.5, 0.999))

n_epochs = 20

for epoch in range(n_epochs):
    for i, data in enumerate(dataloader, 0):


        # update the weights of discriminator
        net_dis.zero_grad()

        # training the discriminator with real images
        real_img, _ = data
        input_x = Variable(real_img)
        target = Variable(torch.ones(input_x.size()[0]))
        output = net_dis(input_x)
        err_dis_real = criterion(output, target)

        # training the discriminator with fake images generated by generator
        noise = Variable(torch.randn(input_x.size()[0], 100, 1, 1))
        fake_img = net_gen(noise)
        target = Variable(torch.zeros(input_x.size()[0]))
        output = net_dis(fake_img.detach())
        err_dis_fake = criterion(output, target)

        # backpropagating the total error
        error_dis = err_dis_real + err_dis_fake
        error_dis.backward()
        optimizer_dis.step()

        # update the weights of generator
        net_gen.zero_grad()
        target = Variable(torch.ones(input_x.size()[0]))
        output = net_dis(fake_img)
        error_gen = criterion(output, target)
        error_gen.backward()
        optimizer_gen.step()

        # print the loss, save the generated images
        print('[{}/{}][{}/{}] Loss_Discriminator: {} Loss_Generator: {}'.format(epoch, n_epochs, i, len(dataloader),
                                        error_dis.item(), error_gen.item()))

        if i % 100 == 0:
            vutils.save_image(real_img, '%s/real_samples.png' % "./results", normalize=True)
            fake = net_gen(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)

