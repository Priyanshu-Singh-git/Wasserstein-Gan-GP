# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:11:32 2024

@author: Priyanshu singh
"""
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch = 60
class DataPrep():
    def __init__(self):
        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,))
                    ])

        self.trainset = torchvision.datasets.MNIST(root = "./data",train=True,transform=self.transform,download=True)

        self.trainloader = torch.utils.data.DataLoader(self.trainset,batch_size=batch,shuffle=True,num_workers=0)
    
    def show_img(self):
        for (data,label) in (self.trainloader):
            data = data.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            fig,axes = plt.subplots(6,10,figsize = (10,10))
            axes = axes.flatten()
            for i in range(batch):
                img = data[i]
                img = np.squeeze(img)
                axe = axes[i]
                axe.imshow(img, cmap='gray')
                axe.set_title(f'Label {label[i]}' , fontsize = 10)
                axe.axis('off')
            plt.tight_layout(pad=0.2)
            plt.show()
            print(i)
            break
            
            
            
dataload = DataPrep()
dataload.show_img()
#%%
def conv_size(input,kernel,stride,padding):
    return int(((input-kernel+(2*padding))/stride)+1)

def deconv_size(input,kernel,stride,padding):
    return int(((input-1)*stride)-(2*padding)+kernel)



#%%


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
           
            nn.utils.spectral_norm(nn.Linear(z_dim, 7*7*256, bias=False)),
            
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 7, 7)),

            
            nn.utils.spectral_norm(nn.ConvTranspose2d(256, 512, kernel_size=5, stride=1, padding=2, bias=False)),
            
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)),
            
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

    

    
noise = torch.randn((batch,100),device = device)
gen = Generator().to(device)
output1 = gen(noise)
print(output1.size())
   #%% 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            
            nn.utils.spectral_norm(nn.Conv2d(1, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

      
            nn.Flatten(),
            nn.Linear(7*7*256, 1)
        )

    def forward(self, x):
        return self.model(x)
        
disc = Discriminator().to(device)
output = disc(output1)
print(output.size())
#%%
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    
        
gen.apply(weights_init)
disc.apply(weights_init)


num_epochs = 25
    
G_loss = []
C_loss = []

    

gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.00030, betas=(0.5, 0.9))
crit_optimizer = torch.optim.Adam(disc.parameters(), lr=0.00020, betas=(0.5, 0.9))  

for epoch in range(num_epochs):
    j = 0
    for image,label in dataload.trainloader:
        batchsize = image.size(0)
        print(batchsize)
        noise = torch.randn((batchsize,100),device = device)
        image = image.to(device)
        for i in range(5):
            fake_img = gen(noise)
            fake_score = disc(fake_img)
            
            real_score = disc(image)
            
            epsilon = torch.rand((batchsize,1,28,28),device=device)
            
            interpolated = (image*epsilon)+(fake_img*(1-epsilon))
            
            score = disc(interpolated)
            gradient = torch.autograd.grad(outputs=score,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones_like(score),
                                           create_graph=True,
                                           retain_graph=True)[0]
            gradient = gradient.view(gradient.shape[0],-1)
            grad_norm =gradient.norm(2,dim=1)
            gp = torch.mean((grad_norm-1)**2)
            disc.zero_grad()
            disc_loss = (torch.mean(fake_score)-torch.mean(real_score)+10*gp)
            disc_loss.backward()
            crit_optimizer.step()
        fake_img = gen(noise)
        fake_score=disc(fake_img)
        gen.zero_grad()
        
        gen_loss = (-1*torch.mean(fake_score))
        gen_loss.backward()
        gen_optimizer.step()
        df = disc_loss.item()
        gf = gen_loss.item()
        gpf = gp.item()
        if j%1000==0:
            with torch.no_grad():
                nois = torch.randn((batchsize,100),device=device)
                img = gen(nois)[0].detach().cpu().permute(1,2,0).numpy()
            
                plt.imshow(img, cmap="gray")
                plt.show()
        
        elif j%100 ==0:    
            
            G_loss.append(gen_loss.item())
            C_loss.append(disc_loss.item())
        print(f"[Epoch{epoch}]- crit:{df:.3f} gen:{gf:.3f} batchno. :{j} gp:{gpf:.3f}")
        j+=100
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss ")
    plt.plot(G_loss,label="G")
    plt.plot(C_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    G_loss.clear()
    C_loss.clear()
#%%
PATH = "D:/pythonProject/WGAN-GP/gener.pth"
Path2 = "D:/pythonProject/WGAN-GP/discr.pth"

# Save
torch.save(gen.state_dict(), PATH)
torch.save(disc.state_dict(),Path2)
#%%
gener = Generator().to(device)
weights = torch.load('D:/pythonProject/WGAN-GP/gener.pth')

gener.load_state_dict(weights)
with torch.no_grad():
    noise = torch.randn((100,100),device=device)
output = gener(noise).detach().cpu()
fig,axes = plt.subplots(10,10,figsize = (60,60))
axes = axes.flatten()
for i in range(100):
    img = output[i].permute(1,2,0).numpy()
    axe = axes[i]
    axe.imshow(img,cmap='gray')
    axe.set_title(f'{i}' , fontsize = 1)
    axe.axis('off')
plt.tight_layout()
plt.show()
    

    
    
        
        
        
        
        
        


    
    
    
    
    
    
    
    
    
    
    

        
