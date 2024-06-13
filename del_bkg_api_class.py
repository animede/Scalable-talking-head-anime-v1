import os
import cv2
from PIL import Image
import torch
import numpy as np
from torch.cuda import amp
from datetime import datetime
from train import AnimeSegmentation
from starlette.responses import Response
from io import BytesIO

class DeleteBackground():
       def __init__(self):
              # イニシャライズ
              self.img_size=1024
              device = torch.device('cpu')
              self.model = AnimeSegmentation.try_load('isnet_is', 'isnetis.ckpt', 'cpu')
              self.model.eval()
              self.model.to(device)

       def del_bkg(self,image, mode):
              print("mode=",mode)
              img = image
              out_img , mask = self.del_bkg_out(img ,mode)
              return out_img , mask

       def del_bkg_out(self,img , img_mode): #del_bkg_out  背景削除     # Input :  img=image , img_mode="pil" or "cv"
               if  img_mode=="pil":
                   img= np.array( img, dtype=np.uint8)
               else:
                   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#カラーチャンネル変換
               mask = self.get_mask(self.model, img , 1024) # mask
               img = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8) # イメージにマスクを追加mask
               pil_img= Image.fromarray(img)
               if  img_mode=="pil":
                   return pil_img , mask  #imgはpillow、maskはcv2
               else:
                   new_image = np.array(pil_img, dtype=np.uint8)
                   img = cv2.cvtColor(new_image , cv2.COLOR_RGBA2BGRA)#opencv形式
                   return img , mask  #imgとmaskはcv2
        
       #+++++++++++++++++++ infference  ++++++++++++++++++++
       def get_mask(self,model, input_img,  s=640):
           h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
           if h > w:
               h, w = s, int(s * w / h)
           else:
               h, w = int(s * h / w), s
           ph, pw = s - h, s - w
           tmpImg = np.zeros([s, s, 3], dtype=np.float32)
           tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
           tmpImg = tmpImg.transpose((2, 0, 1))
           tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
           with torch.no_grad():
               pred = model(tmpImg)
               pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
               pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
               return pred
