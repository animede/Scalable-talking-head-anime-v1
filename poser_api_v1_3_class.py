import torch
import numpy as np
import cv2
from PIL import Image
import time

from tha3.util import extract_pytorch_image_from_PIL_image,torch_linear_to_srgb
from tha3.poser.modes.load_poser import load_poser

#generation Classのバリエーション
#
# inference(self,input_img,current_pose):                    #pose=リポジトリの形式、イメージは毎回ロード
# inference_img(self,current_pose,img_number,user_id):       # pose=リポジトリの形式  イメージは事前ロード,複数画像対応
# inference_pos(self,packed_current_pose,img_number,user_id):# pose=パック形式　イメージは事前ロード,複数画像対応
# inference_dic(self,current_dic,img_number,user_id):        # pose=Dict形式 イメージは事前ロード,複数画像対応

# ユーティリティClass
# get_pose(self,pose_pack):        #パック形式 =>リポジトリの形式変換
# get_init_dic(self):              #Dict形式の初期値を得る
# get_pose_dic(self,dic):          #Dict形式 => リポジトリの形式変換
# load_img(self,input_img,user_id):# 画像をVRAMへ登録

class TalkingHeadAnimeface():
    def __init__(self):

        MODEL_NAME = "separable_float"
        DEVICE_NAME = "cuda"
        self.device = torch.device(DEVICE_NAME)

        self.poser = load_poser(MODEL_NAME, DEVICE_NAME)
        self.poser.get_modules()
        
        self.torch_base_image_list=[0]*20
        self.user_id_list=[0]*20
        self.next_img=0

    def get_pose(self,pose_pack):
        #-----パック形式
        #0  eyebrow_dropdown: str :            "troubled", "angry", "lowered", "raised", "happy", "serious"
        #1  eyebrow_leftt, eyebrow_right:      float:[0.0,0.0]
        #2  eye_dropdown: str:                 "wink", "happy_wink", "surprised", "relaxed", "unimpressed", "raised_lower_eyelid"
        #3  eye_left, eye_right :              float:[0.0,0.0]
        #4  iris_small_left, iris_small_right: float:[0.0,0.0]
        #5 iris_rotation_x, iris_rotation_y : float:[0.0,0.0]
        #6  mouth_dropdown: str:               "aaa", "iii", "uuu", "eee", "ooo", "delta", "lowered_corner", "raised_corner", "smirk"
        #7  mouth_left, mouth_right :          float:[0.0,0.0]
        #8  head_x, head_y :                   float:[0.0,0.0]
        #9  neck_z,                            float
        #10 body_y,                            float
        #11 body_z:                            float
        #12 breathing:                         float
        #
        # Poseの例
        # pose=["happy",[0.5,0.0],"wink", [i/50,0.0], [0.0,0.0], [0.0,0.0],"ooo", [0.0,0.0], [0.0,i*3/50],i*3/50, 0.0, 0.0, 0.0]
        
        pose=[float(0)]*45
        #eyebrow
        eyebrow=["troubled", "angry", "lowered", "raised", "happy", "serious"]
        index=2*(eyebrow.index(pose_pack[0]))#eyebrowの指定からposeリストの位置を計算
        pose[index]=float(pose_pack[1][0])
        pose[index+1]=float(pose_pack[1][1])
        #print("eyebrow=",eyebrow[int(index/2)],pose[index],pose[index+1])
        #eye
        eye=["wink", "happy_wink", "surprised", "relaxed", "unimpressed", "raised_lower_eyelid"]
        index=2*(eye.index(pose_pack[2]))#eyeの指定からposeリストの位置を計算
        pose[12+index]=float(pose_pack[3][0])
        pose[12+index+1]=float(pose_pack[3][1])
        #iris
        pose[24]=float(pose_pack[4][0])
        pose[25]=float(pose_pack[4][1])
        #print("iris_small=",pose[25],pose[26])
        #iris_rotation
        pose[37]=float(pose_pack[5][0])
        pose[38]=float(pose_pack[5][1])
        print("iris_rotation=",pose[37],pose[38])   
        #mouth
        mouth=["aaa", "iii", "uuu", "eee", "ooo", "delta", "lowered_corner", "raised_corner", "smirk"]
        index=2*(mouth.index(pose_pack[6]))#mouthの指定からposeリストの位置を計算
        pose[26+index]=float(pose_pack[7][0])
        pose[26+index+1]=float(pose_pack[7][1])
        #print("mouth=",mouth[int(index/2)],pose[26+index],pose[26+index+1])      
        #head
        pose[39]=float(pose_pack[8][0])
        pose[40]=float(pose_pack[8][1])
        #print("head_x,y=",pose[39],pose[40])
        #neck
        pose[41]=float(pose_pack[9])
        #body_y
        pose[42]=float(pose_pack[10])
        #body_z
        pose[43]=float(pose_pack[11])
        #breathing
        pose[44]=float(pose_pack[12])
        #print("neck=",pose[41],"body_y=",pose[42],"body_z=",pose[43],"breathing=",pose[44])

        return pose

    def get_init_dic(self):
        pose_dic_org={"eyebrow":{"menue":"happy","left":0.0,"right":0.0},
              "eye":{"menue":"wink","left":0.0,"right":0.0},
              "iris_small":{"left":0.0,"right":0.0},
              "iris_rotation":{"x":0.0,"y":0.0},
              "mouth":{"menue":"aaa","val":0.0},
              "head":{"x":0.0,"y":0.0},
              "neck":0.0,
              "body":{"y":0.0,"z":0.0},
              "breathing":0.0,
              }
        return pose_dic_org
        

    def get_pose_dic(self,dic):
            #サンプル Dict形式
            #"mouth"には2種類の記述方法がある"lowered_corner"と”raised_corner”は左右がある
            #  "mouth":{"menue":"aaa","val":0.0},
            #  "mouth":{"menue":"lowered_corner","left":0.5,"right":0.0},　これはほとんど効果がない
            #
            #pose_dic={"eyebrow":{"menue":"happy","left":0.5,"right":0.0},
            #        "eye":{"menue":"wink","left":0.5,"right":0.0},
            #        "iris_small":{"left":0.0,"right":0.0},
            #        "iris_rotation":{"x":0.0,"y":0.0},
            #        "mouth":{"menue":"aaa","val":0.7},
            #        "head":{"x":0.0,"y":0.0},
            #        "neck":0.0,
            #        "body":{"y":0.0,"z":0.0},
            #        "breathing":0.0
            #        }
            
        pose=[float(0)]*45
        #eyebrow
        eyebrow=["troubled", "angry", "lowered", "raised", "happy", "serious"]
        eyebrow_menue=dic["eyebrow"]["menue"]
        index=2*(eyebrow.index(eyebrow_menue))#eyebrowの指定からposeリストの位置を計算
        pose[index]=float(dic["eyebrow"]["left"])
        pose[index+1]=float(dic["eyebrow"]["right"])
        #eye
        eye=["wink", "happy_wink", "surprised", "relaxed", "unimpressed", "raised_lower_eyelid"]
        eye_menue=dic["eye"]["menue"]
        index=2*(eye.index(eye_menue))#eyeの指定からposeリストの位置を計算
        pose[12+index]=float(dic["eye"]["left"])
        pose[13+index]=float(dic["eye"]["right"])
        #iris_small
        pose[24]=float(dic["iris_small"]["left"])
        pose[25]=float(dic["iris_small"]["right"])
        #iris_rotation
        pose[37]=float(dic["iris_rotation"]["x"])
        pose[38]=float(dic["iris_rotation"]["y"])
        #mouth
        mouth=["aaa", "iii", "uuu", "eee", "ooo", "delta","lowered_corner","raised_corner", "smirk"]
        mouth_menue=dic["mouth"]["menue"]
        if mouth_menue== "lowered_corner": #"lowered_corner","lowered_corner"
            pose[32]=float(dic["mouth"]["left"])             
            pose[33]=float(dic["mouth"]["right"])
        elif mouth_menue== "raised_corner":  #"lowered_corner","raised_corner"
            pose[34]=float(dic["mouth"]["left"])             
            pose[35]=float(dic["mouth"]["right"])   
        elif mouth_menue== "smirk":    #"smirk"
            pose[36]=float(dic["mouth"]["val"])             
        else:
            index=(mouth.index(mouth_menue)) #"aaa", "iii", "uuu", "eee", "ooo", "delta"
            pose[26+index]=float(dic["mouth"]["val"]) 
        #head
        head_x=dic["head"]["x"]
        head_y=dic["head"]["y"]           
        pose[39]=float(head_x)
        pose[40]=float(head_y)
        #neck
        pose[41]=float(dic["neck"])
        #body
        body_y=dic["body"]["y"]
        body_z=dic["body"]["z"]           
        pose[42]=float(body_y)
        pose[43]=float(body_z)
        #breathing
        pose[44]=float(dic["breathing"])
        return pose

    def load_img(self,input_img,user_id):
        if self.next_img>19:
            self.next_img=0
        img_number=self.next_img
        if user_id>19:
            img_number=-1
            return img_number #Error
        self.user_id_list[img_number]=user_id
        self.next_img +=1
        if self.next_img>19:
            self.next_img=0
        self.torch_base_image_list[img_number]=extract_pytorch_image_from_PIL_image(input_img).cuda(0)
        return img_number

    def inference(self,input_img,current_pose,out="pil"):#リポジトリの形式-オリジナルポーズ形式、イメージは毎回ロード
        torch_base_image = extract_pytorch_image_from_PIL_image(input_img).cuda(0)
        pose = torch.tensor(current_pose, dtype=self.poser.get_dtype()).cuda(0)
        with torch.inference_mode():
            output_image = self.poser.pose(torch_base_image, pose)[0]
        return out_image_form(output_image, out)
    
    def inference_img(self,current_pose,img_number,user_id,out="pil"):# リポジトリの形式-オリジナルポーズ形式  イメージは事前ロード
        if self.user_id_list[img_number]!=user_id:
            return 0 #image data was updated
        pose = torch.tensor(current_pose, dtype=self.poser.get_dtype()).cuda(0)
        with torch.inference_mode():
            output_image = self.poser.pose(self.torch_base_image_list[img_number], pose)[0]
        return out_image_form(output_image, out)

    def inference_pos(self,packed_current_pose,img_number,user_id,out="pil"):#　　パック形式　イメージは事前ロード
        if self.user_id_list[img_number]!=user_id:
            return 0 #image data was updated
        current_pose=self.get_pose(packed_current_pose) 
        output_image = self.inference_img(current_pose,img_number,user_id,out)
        return output_image

    def inference_dic(self,current_dic,img_number,user_id,out="pil"):#　Dict形式 イメージは事前ロード
        if self.user_id_list[img_number]!=user_id:
            return 0 #image data was updated
        current_pose=self.get_pose_dic(current_dic)#current_dic==>current_pose2
        output_image=self.inference_img(current_pose,img_number,user_id,out)
        return output_image
    
# internal function
def out_image_form(in_image, out):
    if out=="pil":
        image = convert_output_image_from_torch_to_pil(in_image)
    elif out=="cv2":
        image = convert_output_image_from_torch_to_cv2(in_image)
    else:
        image = in_image #Tenser
    return image
    
def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

def convert_output_image_from_torch_to_pil(output_image):
    output_image = output_image.float()
    output_image = convert_linear_to_srgb((output_image + 1.0) / 2.0)
    c, h, w = output_image.shape
    output_image = 255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
    output_image = output_image.byte()
    numpy_image = output_image.detach().cpu().numpy()
    return Image.fromarray(numpy_image[:, :, 0:4], mode="RGBA")

def convert_output_image_from_torch_to_cv2(output_image):
    output_image = output_image.float()
    output_image = convert_linear_to_srgb((output_image + 1.0) / 2.0)
    c, h, w = output_image.shape
    output_image = 255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
    output_image = output_image.byte()
    numpy_image = output_image.detach().cpu().numpy()
    imge = cv2.cvtColor(numpy_image, cv2.COLOR_RGBA2BGRA)
    return imge

