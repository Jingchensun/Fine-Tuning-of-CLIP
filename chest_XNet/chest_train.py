
#######Chest-Xay train
import numpy as np
import torch
import clip
from imagenetv2_pytorch import ImageNetV2Dataset
import torchvision

from read_data import ChestXrayDataSet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import os
import torch
import glob
from PIL import Image
import random
import clip
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def zeroshot_classifier(model):
    classnames = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    templates = ['a photo of a {}.']
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            #print(texts)
            texts = clip.tokenize(texts).cuda() #tokenize
            #print(texts.size()) #torch.Size([80, 77])
            class_embeddings = model.encode_text(texts) #embed with text encoder
            # print(class_embeddings.size()) #torch.Size([80, 512])
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            #print('1',class_embeddings.size()) # torch.Size([80, 512])
            class_embedding = class_embeddings.mean(dim=0)
            #print('2',class_embedding.size()) #torch.Size([512])
            class_embedding /= class_embedding.norm()
            #print('3',class_embedding.size()) #torch.Size([512])
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 



def train(model, trainloader, testloader, zeroshot_weights, learning_rate):
    max_AUROC_avg = 0.5
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    for epoch in range(EPOCH):
        print('epoch:', epoch)
        loss_record = []
        for batch in tqdm(trainloader):
            optimizer.zero_grad()
            list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images
            # print(list_image.size()) #torch.Size([32, 3, 224, 224])
            
            images = torch.tensor(np.stack(list_image)).to(device)
            texts = clip.tokenize(list_txt).to(device) #torch.Size([32, 77])
                # print(texts.size()) #torch.Size([32, 77])
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(BATCH_SIZE,dtype=torch.long,device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            #print('total loss:', total_loss)
            
            convert_models_to_fp32(model)
            optimizer.step()
            loss_record.append(total_loss.item())
            clip.model.convert_weights(model)
        loss_train_mean = np.mean(loss_record)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % 10 == 0 and epoch != 0:
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            torch.save(model.state_dict(),
                        os.path.join(save_model_path, 'latest_{}.pth_%'.format(epoch)))

        if epoch % 5 ==0:
            AUROC_avg = val(model,testloader,zeroshot_weights)
            if AUROC_avg > max_AUROC_avg:
                max_AUROC_avg = AUROC_avg
                torch.save(model.state_dict(),
                           os.path.join(save_model_path, 'best_{}_epoch{}.pth'.format(max_AUROC_avg,epoch)))
def val(model, testloader, zeroshot_weights):
    print('start val!')
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    with torch.no_grad():
        #top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(testloader)):
            images = images.cuda()
            #print('image',images.size()) #image torch.Size([32, 3, 224, 224])
            target = target.cuda()
            #print('target:',target.size()) #target: torch.Size([32, 14])
            gt = torch.cat((gt, target), 0)
            #print(target.size()) #torch.Size([32])
            
            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights
            # print('image_features',image_features.size()) #torch.Size([32, 512])
            # print('zero_shot', zeroshot_weights.size()) #torch.Size([512, 1000])
            # print('logits:', logits.size()) #torch.Size([32, 1000])
            # image_features torch.Size([32, 512])
            # zero_shot torch.Size([512, 14])
            # logits: torch.Size([32, 14])
            pred = torch.cat((pred, logits.data), 0)

        print(gt.size(),pred.size())
        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))

        return AUROC_avg
                
def compute_AUCs(gt, pred):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(14):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

EPOCH =100
BATCH_SIZE =128
learning_rate = 5e-5

save_model_path='clip_chest/'
zeroshot_weights = zeroshot_classifier(model)

train_dataset = ChestXrayDataSet(data_dir='/home/jason/data/chestx_ray14/images/images',
                                image_list_file='/home/jason/CheXNet/ChestX-ray14/labels/test_list.txt',
                                mode='train',
                                transform=preprocess
                                # transform=transforms.Compose([
                                #     transforms.Resize(256),
                                #     transforms.TenCrop(224),
                                    # ])
                                    )
trainloader = DataLoader(   dataset=train_dataset, 
                            batch_size=BATCH_SIZE,
                            shuffle=True, 
                            num_workers=8, 
                            pin_memory=True,
                            drop_last=True)

test_dataset = ChestXrayDataSet(data_dir='/home/jason/data/chestx_ray14/images/images',
                                image_list_file='/home/jason/CheXNet/ChestX-ray14/labels/test_list.txt',
                                mode='test',
                                transform=preprocess
                                    )
testloader = DataLoader(    dataset=test_dataset, 
                            batch_size=BATCH_SIZE,
                            shuffle=True, 
                            num_workers=8, 
                            pin_memory=True,
                            drop_last=True)


train( model, trainloader, testloader, zeroshot_weights, learning_rate)