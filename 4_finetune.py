#new numpy based clip
import os
import torch
import glob
from PIL import Image
import random
import clip
from tqdm.notebook import tqdm
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm


from clip.model import Gauss_model
from tqdm.notebook import tqdm

device = "cuda:3" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model_g = Gauss_model().to(device)
_, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training


EPOCH =10
BATCH_SIZE =32

class cocodtrain(torch.utils.data.Dataset):
    def __init__(self, image_path='/home/jason/data/coco/images', text_path='/home/jason/data/coco/text', mode='train2014'):

        self.image_list = []
        self.image_list.extend(glob.glob(os.path.join(image_path, mode, '*.jpg')))
        self.image_list.sort()

        self.label_list = []
        self.label_list.extend(glob.glob(os.path.join(text_path, mode, '*.txt')))
        self.label_list.sort()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        image = image.resize((224,224), Image.BILINEAR)
        image = preprocess(image)
        #image = np.asarray(image)

        with open(self.label_list[index], "r") as f:
            data = f.readlines()
            label = random.choice(data)
            
        return image, label

trainset = cocodtrain('/home/jason/data/coco/images','/home/jason/data/coco/text','train2014')
trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=BATCH_SIZE,
                    shuffle=True, 
                    num_workers=0,
                    drop_last=True)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_g.parameters(), lr=5e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset


def loss_multivariate_normal_kl2(mu_1, sigmasquare_1, mu_2, sigmasquare_2):
    b, dim = mu_1.shape
    kl = 0
    kkll = torch.ones(b,b)
    for x in range(b):
        for y in range(b):
            c_mu_1 = mu_1[x]
            c_Sigma_1 = torch.diag(sigmasquare_1[x])

            c_mu_2 = mu_2[y]
            c_Sigma_2 = torch.diag(sigmasquare_2[y])

            p1 = torch.prod(sigmasquare_2)
            p2 = torch.prod(sigmasquare_1)
            if p1 == 0 or p2 ==0:
                first = 0
            else:
                first = p1.log() - p2.log()

            #first = c_Sigma_2.det().log() - c_Sigma_1.det().log()
            second = -dim
            third = torch.matmul(c_Sigma_2.inverse(), c_Sigma_1).trace()
            fourth = torch.matmul(torch.matmul((c_mu_2 - c_mu_1).T, c_Sigma_2.inverse()), c_mu_2 - c_mu_1)
            kl = 0.5 * (first + second + third + fourth)
            kkll[x,y] = kl
    return kkll

n = BATCH_SIZE
Kl_matric = np.ones([n,n])

for epoch in range(EPOCH):
    print('epoch:', epoch)
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
        # print(list_image.size()) #torch.Size([32, 3, 224, 224])
        #print(len(list_txt))
        images = torch.tensor(np.stack(list_image),requires_grad=True).to(device)
        #print('image size:',images.size()) #image size: torch.Size([32, 3, 224, 224])
        texts = clip.tokenize(list_txt).to(device) #torch.Size([32, 77])
        #  print(texts.size()) #torch.Size([32, 77])
        image_u,image_std,text_u,text_std= model_g(images, texts)
        #print(image_u.size()) #torch.Size([32, 512])

        p_q_kl = loss_multivariate_normal_kl2(image_u,image_std,text_u,text_std)
        #print(p_q_kl.size())
        # print(Kl_matric.requires_grad)
        logits_per_image = torch.tensor(p_q_kl,requires_grad=True).to(device)
        logits_per_text = logits_per_image.t()
        #print(logits_per_image.size(),logits_per_text.size())

        ground_truth = torch.arange(BATCH_SIZE,dtype=torch.long,device=device)
        #print(ground_truth.size())
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        print('total loss:', total_loss)
      
        #convert_models_to_fp32(model)
        optimizer.step()
        #clip.model.convert_weights(model)

torch.save({
    'epoch': epoch,
    'model_state_dict': model_g.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': total_loss,
    }, f"model_checkpoint/model_10.pt") #just change to your preferred folder/filename   