import time
import os
import sys
import numpy as np
from PIL import Image
import torch
import requests

import shutil
import glob
import pickle
import cv2

from  face_d_api_class import AnimeFaceDetect
from tkh_up_scale import upscale
from del_bkg_api_class import DeleteBackground

class Image2form():
    def __init__(self,up_url):
        self.up_url=up_url
        self.AF=AnimeFaceDetect()
        self.BG=DeleteBackground()
        
    # *************** TKH用にinput_imageをフォーミングすす。out_formで出力をpilかcvに指定、mask：True= マスク生成,del_flegで背景削除の有無を指定する。
    #    例）Talking-Head用アライメント画像作成
    #        input_image =image_data_form(input_image ,"pil",False)

    def image_data_form(self, input_image,out_form="pil",mask=False,del_flag=True): #入出力の画像形式はout_formで指定
        result =True
        if out_form=="cv":#input is cv
            height, width, channels = input_image.shape
            if channels != 4: #αチャンネルがなければ背景を削除してチャンネル追加
                input_image,_=self.BG.del_bkg_out(input_image , "cv")
        else: 
            if input_image.mode != "RGBA": #αチャンネルがな背景を削除してチャンネル追加
                input_image ,_=self.BG.del_bkg_out(input_image , "pil")
            np_w_img = np.array(input_image, dtype=np.uint8)
            input_image = cv2.cvtColor(np_w_img, cv2.COLOR_RGBA2BGRA) #input_image = 背景を削除 OpeCV
        cv_w_img = cv2.cvtColor(input_image, cv2.COLOR_BGRA2BGR)#Face detectのためにαチャンネルを削除
        imge, dnum, predict_bbox, pre_dict_label_index, scores =self.AF.face_det_head(cv_w_img,1.68,0.5, 0.5)#face-head検出   ratio=1.68, shift=0.5, confidence_level=0.5
        print("bbox=",predict_bbox,"label=",pre_dict_label_index,"score=",scores)
        try:
            box=predict_bbox[0]
            print("box= ",box)
            #face-head検出のバウンディングbox付きの大きさを元に画像の拡大率を計算（THKのフォフォームに合わせるため：Head=128標準 ）
            box_disp=(box[0],box[1]),(box[2],box[3])
            print(box_disp)
            print("BOX SIZE=",int(box[0]-box[2]),int(box[1]-box[3]))
            resize_facter=128/int(box[0]-box[2])
            print("resize_facter=",resize_facter) #HeadからResizeのファクタを計算
        except:
            result = "resize error" #resize_facteの計算が正しく行えなかった場合は1=なにもしない。
            return result
            
        if resize_facter > 4:   #2倍以上の拡大は推奨できないのでエラー
            print("image is too small")
            result="image is too small"
        elif resize_facter > 2: #4倍して所定のサイズに縮小する
            input_image = upscale(self.up_url ,input_image, "full", 4) #upscale
            image_resaize=resize_facter/4
        elif resize_facter > 1: #2倍して所定のサイズに縮小する
            input_image = upscale(self.up_url ,input_image, "full", 2) #upscale
            image_resaize=resize_facter/2
        else: # 1> reasize >0 なのでそのまま縮小率として使う
            image_resaize=resize_facter
            
        height, width, channels = input_image.shape
        cv_resize_img = cv2.resize(input_image, dsize=(round(width*image_resaize),round(height*(image_resaize))),interpolation = cv2.INTER_AREA)
        height, width, channels = cv_resize_img.shape#縮小した画像のH,W取得
        print("resize_image h= ",height,"w= ",width,"Channels= ",channels)
        #バウンディングboxは検出字のresize_facterを使う
        top=int(box[3]*resize_facter)
        left=int(box[2]*resize_facter)
        bottom=int(box[1]*resize_facter)
        right=int(box[0]*resize_facter)    
        print("top=",top,"left=",left,"bottom=",bottom,"right=",right)  
        #αチャンネル付き入力画像をPILイメージに変換
        nd_input_image = cv2.cvtColor(cv_resize_img, cv2.COLOR_BGRA2RGBA)
        pil_input_image = Image.fromarray(nd_input_image)
        # 512x512ピクセルで、全てのピクセルが透明な画像を作成
        pil_out_image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        # ペーストする位置を指定
        p_top = 64-top #バウンディングboxの位置が64pixよりも大きければ差がペースト位置、小さければ差がマイナスになっているがpilではOK
        p_left =192-left 
        # 画像の大きさを調整した入力画像をアルファチャンネルを考慮して前景画像を背景画像にペースト
        pil_out_image.paste(pil_input_image, (p_left, p_top), pil_input_image)
        #pil_out_image.show()
        if out_form=="pil":
            return result, pil_out_image
        elif out_form=="cv":
            np_w_img = np.array(pil_out_image, dtype=np.uint8)
            cv_out_image = cv2.cvtColor(np_w_img, cv2.COLOR_RGBA2BGRA) #input_image = 背景を削除 OpeCV
            return result, cv_out_image



    

