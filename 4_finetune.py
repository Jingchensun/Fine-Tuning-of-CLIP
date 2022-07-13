#torch based model_g clip
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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs2')

import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel 


from clip.model import Gauss_model
from tqdm.notebook import tqdm

# parser = argparse.ArgumentParser(description='Network Parser')
# args = parser.parse_args()
# local_rank = args.local_rank

# torch.cuda.set_device(local_rank)



# # prepocess
# try:
#     from torchvision.transforms import InterpolationMode
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
#     BICUBIC = Image.BICUBIC

# def _convert_image_to_rgb(image):
#     return image.convert("RGB")

# def _transform(n_px):
#     return Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])
# preprocess = _transform([224,224])

#device = "cuda:0" if torch.cuda.is_available() else "cpu" 

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
model_g = nn.DataParallel(Gauss_model()).cuda()#.to(device)
#_, preprocess = clip.load("ViT-B/32",device="cuda:0",jit=False) #Must set jit=False for training

_, preprocess = clip.load("ViT-B/32",jit=False) #Must set jit=False for training


EPOCH =50
BATCH_SIZE =64
checkpoint_step = 5
validation_step = 9

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


def loss_multivariate_normal_kl_multi(mu_1, sigma_1, mu_2, sigma_2):
    b, dim = mu_1.shape
    #print(b,dim)
    mu_1 = mu_1.unsqueeze(1).unsqueeze(2)
    sigmasquare_1 = sigma_1.unsqueeze(1)
    mu_2 = mu_2.unsqueeze(0).unsqueeze(2)
    sigmasquare_2 =sigma_2.unsqueeze(0)

    p1 = torch.prod(sigmasquare_1,-1)
    p2 = torch.prod(sigmasquare_2,-1)
    if torch.any(p1 == 0) or torch.any(p2 == 0):
        first = 0
        #print('p1=======0')
    else:
        first = p2.log() - p1.log()

    #first = torch.log(torch.prod(sigmasquare_2,-1)/torch.prod(sigmasquare_1,-1))
    #print('first',first)
    second = -dim
    #print('second',second)
    third = ((sigmasquare_1 / sigmasquare_2)).sum(-1)
    #print('third',third)
    a = mu_1 - mu_2
    b = torch.matmul(a, torch.diag_embed(1/sigmasquare_2))
    c = torch.transpose(a,3,2)
    fourth = torch.matmul(b, c).squeeze(3).squeeze(2)
    
    kl = 0.5 * (first + second + third + fourth)
    return kl

n = BATCH_SIZE
Kl_matric = np.ones([n,n])
step = 0
for epoch in range(EPOCH):
    print('epoch:', epoch)
    loss_record = []
    for batch in tqdm(trainloader):
       
        list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
        # print(list_image.size()) #torch.Size([32, 3, 224, 224])
        #print(len(list_txt))
        images = torch.tensor(np.stack(list_image),requires_grad=True).cuda()#to(device)
        #print('image size:',images.size()) #image size: torch.Size([32, 3, 224, 224])
        texts = clip.tokenize(list_txt).cuda()#to(device) #torch.Size([32, 77])
        #  print(texts.size()) #torch.Size([32, 77])
        image_u,image_std,text_u,text_std= model_g(images, texts)
        #print(image_u.size()) #torch.Size([32, 512])

        p_q_kl = loss_multivariate_normal_kl_multi(image_u,image_std,text_u,text_std)
        #print(p_q_kl.size())
        # print(Kl_matric.requires_grad)
        logits_per_image = torch.tensor(p_q_kl,requires_grad=True).cuda()#to(device)
        logits_per_text = logits_per_image.t()
        #print(logits_per_image.size(),logits_per_text.size())

        #ground_truth = torch.arange(BATCH_SIZE,dtype=torch.long,device=device)
        ground_truth = torch.arange(BATCH_SIZE,dtype=torch.long).cuda()

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        #print('total_loss',total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        step+=1
        writer.add_scalar("step loss",total_loss,step)
        loss_record.append(total_loss.item())

    loss_train_mean = np.mean(loss_record)
    writer.add_scalar('epoch_loss_train', float(loss_train_mean), epoch)
    print('loss for train : %f' % (loss_train_mean))

    # if epoch % checkpoint_step == 0 and epoch != 0:
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model_g.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss_train_mean,
    #         }, f"model_checkpoint/modelg_epoch_%s.pt"%epoch) #just change to your preferred folder/filename 

    if epoch % validation_step == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_g.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_train_mean,
            }, f"model_checkpoint/modelg_epoch_%s.pt"%epoch) #just change to your preferred folder/filename  
