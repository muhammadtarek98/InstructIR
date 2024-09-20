import cv2
import torch
import torchvision
import numpy as np
from InstructIR.text.models import LMHead
from metrics import pt_psnr, calculate_ssim, calculate_psnr
from models.instructir import create_model
from text.models import LanguageModel
import yaml
import os
from utils import dict2namespace

def load_image_models(device:torch.device,ckpt_path:str,CONFIG:str):
    with open(os.path.join(CONFIG), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    model = create_model(input_channels=config.model.in_ch,
                                    width=config.model.width,
                                    enc_blks=config.model.enc_blks,
                                    middle_blk_num=config.model.middle_blk_num,
                                    dec_blks=config.model.dec_blks,
                                    txtdim=config.model.textdim)
    ckpt=torch.load(f=ckpt_path,map_location=device,weights_only=True)
    model.load_state_dict(ckpt)
    model=model.to(device)
    return model
def load_language_models(device:torch.device,
                         ckpt_path:str,
                         config_path:str):
    with open(os.path.join(config_path), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    lmodel=config.llm.model
    model=LanguageModel(model=lmodel)
    lm_head=LMHead(embedding_dim=config.llm.model_dim,
                   hidden_dim=config.llm.embd_dim,
                   num_classes=config.llm.nclasses)
    ckpt=torch.load(f=ckpt_path,map_location=device)
    lm_head.load_state_dict(ckpt,strict=True)
    lm_head=lm_head.to(device)
    model=model.to(device)
    return model,lm_head
def img_2_tensor(img):
    transform=torchvision.transforms.Compose(
        transforms=[
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(720,720),
                                          interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
            torchvision.transforms.Normalize(mean=[0,0,0],
                                             std=[1,1,1])])

    raw_image_tensor=transform(img)
    raw_image_tensor=torch.unsqueeze(input=raw_image_tensor,dim=0)
    return raw_image_tensor
def tensor_2_img(tensor):
    print(tensor.shape)
    #tensor=tensor.squeeze(tensor)
    tensor=torch.permute(tensor,dims=(1,2,0))
    tensor=tensor.detach().cpu().numpy()
    arr=np.clip(a=tensor,a_max=1,a_min=0)
    arr=(arr*255.0).astype(np.uint8)
    return arr



image_ckpt_path="models/im_instructir-7d.pt"
language_ckpt_path="models/lm_instructir-7d.pt"
config_path="configs/eval5d.yml"
device=torch.device("cpu")
image_model= load_image_models(device=device,
                               ckpt_path=image_ckpt_path,
                               CONFIG=config_path)

image_model.eval()
l_model,lm_head=load_language_models(device=device,
                                   config_path=config_path,
                                   ckpt_path=language_ckpt_path)
l_model.eval()
"""image=cv2.cvtColor(
    src=
)"""
image=cv2.imread(filename="Image_enhancement/dataset/Enhancement_Dataset/7426_NF2_f000150.jpg")
print(type(image))
input_tensor=img_2_tensor(img=image)
input_tensor=input_tensor.to(device)
print(next(image_model.parameters()).device)

print(next(l_model.parameters()).device)

print(next(lm_head.parameters()).device)
txt="can you remove the water effect"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(input_tensor.device)
lm_embd=l_model(txt).to(device)
lm_embd=lm_embd.to(device)

text_embed,deg_pred=lm_head(lm_embd)
with torch.no_grad():
    prediction=image_model(input_tensor,text_embed)

prediction=prediction.squeeze_()
output_image=tensor_2_img(tensor=prediction)
cv2.imshow("test",output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()