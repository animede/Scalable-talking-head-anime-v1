import numpy as np
import cv2
from PIL import Image
import time
from time import sleep
import requests
import pickle
import multiprocessing
from tkh_up_scale import upscale

#generation Classのバリエーション
#
# inference(self,input_img,current_pose):                    #pose=リポジトリの形式、イメージは毎回ロード
# inference_img(self,current_pose,img_number,user_id):       # pose=リポジトリの形式  イメージは事前ロード,複数画像対応
# inference_pos(self,packed_current_pose,img_number,user_id):# pose=パック形式　イメージは事前ロード,複数画像対応
# inference_dic(self,current_dic,img_number,user_id):        # pose=Dict形式 イメージは事前ロード,複数画像対応
# mp_pose2image_frame # マルチプロセス版　pose→イメージ＋クロップ＋UPSCALE
# mp_pack2image_frame # マルチプロセス版　pose_pack→イメージ＋クロップ＋UPSCALE
# mp_dic2image_frame  # マルチプロセス版　pose_dict→イメージ＋クロップ＋UPSCALE

# ユーティリティClass
# get_pose(self,pose_pack):        #パック形式 =>リポジトリの形式変換
# get_init_dic(self):              #Dict形式の初期値を得る
# get_pose_dic(self,dic):          #Dict形式 => リポジトリの形式変換
# load_img(self,input_img,user_id):# 画像をVRAMへ登録
# create_mp_upscale(self,url)      # upscaleプロセスの開始
# proc_terminate(self)             # upscaleプロセスの停止
# mp_pose2image_frame # マルチプロセス版　pose→イメージ＋クロップ＋UPSCALE
# mp_pack2image_frame # マルチプロセス版　pose_pack→イメージ＋クロップ＋UPSCALE
# mp_dic2image_frame # マルチプロセス版　pose_dict→イメージ＋クロップ＋UPSCALE

class TalkingHeadAnimefaceInterface():
    def __init__(self,host):
        userid=0
        self.url=host#Talking-Head-Animefaceのサーバホスト

        #アップスケールマルチプロセッシングのqueue,process初期化
        self.queue_in_image = None
        self.queue_out_image =None
        self.proc = None
        self.queue_tkh_pose =None
        self.queue_tkh_image =None
        self.tkh_proc =None
        self.previous_image = np.zeros((512, 512, 3), dtype=np.uint8)#upscaleが最初に呼び出される時に画像ができていないので初期値を設定

    def image_2_form(self,image, image_mode="pil"):
        data = {"image_mode": image_mode}
        send_img_data = pickle.dumps(image, 5) 
        files = {"image": ("img.dat",  send_img_data, "application/octet-stream")}
        response = requests.post(self.url + "/image_2_form/", files=files, data=data) #リクエスト送信
        if response.status_code == 200:
            image_data = response.content
            recived_form =(pickle.loads(image_data))#元の形式にpickle.loadsで復元 dnum,rgb_img, predict_bbox, pre_dict_label_index, scores
        return recived_form
        
    def get_init_dic(self):
        response = requests.post(self.url+"/get_init_dic/") #リクエスト
        if response.status_code == 200:
            pose_data = response.content
            org_dic =(pickle.loads(pose_data))#元の形式にpickle.loadsで復元
        return org_dic  
      
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
        
        pose_pack_pkl = pickle.dumps(pose_pack, 5)
        files = {"pose":("pos.dat",pose_pack_pkl, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/get_pose/", files=files) #リクエスト
        if response.status_code == 200:
            pose_data = response.content
            pose =(pickle.loads(pose_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return pose  

    #アップスケールプロセスの開始
    def create_mp_upscale(self,url):
        self.queue_in_image = multiprocessing.Queue()   # 入力Queueを作成
        self.queue_out_image = multiprocessing.Queue()  # 出力Queueを作成
        self.proc = multiprocessing.Process(target=self._mp_upscal, args=(url ,self.queue_in_image,self.queue_out_image))  #process作成
        self.proc.start() #process開始
        pid=self.proc.pid
        return pid  

    #アップスケールプロセス実行関数--terminateされるまで連続で動きます
    def _mp_upscal(self,url ,queue_in_image,queue_out_image):
        print("Process started")
        while True:
            #if self.queue_in_image.empty()==False:
                print("++++@@@+++  _mp_upscal")
                received_data = self.queue_in_image.get() # queue_in_imageからデータを取得
                received_image=received_data[0]      
                mode=received_data[1]
                scale=received_data[2]      
                out_image=upscale(url ,received_image, mode, scale)
                self.queue_out_image.put(out_image)
                time.sleep(0.001)
            #time.sleep(0.001)
            
    #アップスケールプロセス停止関数--terminate
    def up_scale_proc_terminate(self):
        while not self.queue_out_image.empty():
            self.queue_out_image.get_nowait()
        while not self.queue_in_image.empty():
            self.queue_in_image.get_nowait()
        self.proc.terminate()#サブプロセスの終了
        print("Upscale process terminated")

    #tkhプロセスの開始
    def create_mp_tkh(self):
        self.queue_tkh_image = multiprocessing.Queue()   # 入力Queueを作成
        self.queue_tkh_pose = multiprocessing.Queue()  # 出力Queueを作成
        self.tkh_proc = multiprocessing.Process(target=self._mp_tkh, args=(self.queue_tkh_image,self.queue_tkh_pose))  #process作成
        self.tkh_proc.start() #process開始  

    #tkhプロセス実行関数--terminateされるまで連続で動きます
    def _mp_tkh(self,queue_tkh_pose,queue_tkh_image):
        print("Tkh process started")
        while True:
            if self.queue_tkh_pose.empty()==False:
                received_data = self.queue_tkh_pose.get()
                current_pose=received_data[0]      
                img_number=received_data[1]
                user_id=received_data[2]      
                out_typ=received_data[3]    
                result, out_image=self.inference_img(current_pose,img_number,user_id,out=out_typ)
                self.queue_tkh_image.put(out_image)
            time.sleep(0.002)
            
    #tkhプロセス停止関数--terminate
    def tkh_proc_terminate(self):
        while not self.queue_tkh_pose.empty():
            self.queue_tkh_pose.get_nowait()
        while not self.queue_tkh_image.empty():
            self.queue_tkh_image.get_nowait()
        self.tkh_proc.terminate()#サブプロセスの終了
        print("Tkh process terminated")

    #Dict形式ポースデータから画像を生成し、アップスケールまで行う
    # global_out_image  :現在のイメージ
    # current_pose      :動かしたいポーズ
    # img_number        :アップロード時に返送されるイメージ番号
    # user_id           :画像の所有者id
    # mode              :クロップモード　早い＜breastup・waistup・upperbody・full＜遅い
    # scale             :画像の倍率　2/4/8　大きい指定ほど生成が遅くなる
    # fps               :指定フレームレート。生成が指定フレームレートに間に合わない場合は現在イメージ返送される
    def mp_pose2image_frame(self,global_out_image,current_pose,img_number,user_id,mode,scale,fps):
        frame_start=time.time()
        result,out_image=self.inference(current_pose,img_number,user_id,"cv2") 
        up_scale_image,result = self._mp_get_upscale(global_out_image,out_image,mode,scale,fps,frame_start)
        try:
            sleep(1/fps - (time.time()-frame_start))
        except:
            print("Remain time is minus")
        return up_scale_image,result

    def mp_pack2image_frame(self,global_out_image,packed_current_pose,img_number,user_id,mode,scale,fps):
        frame_start=time.time()
        current_pose=self.get_pose(packed_current_pose) #packed_pose=>current_pose
        out_image=self._mp_inference_img(global_out_image, current_pose,img_number,user_id,out_typ="cv2")
        up_scale_image,result = self._mp_get_upscale(global_out_image,out_image,mode,scale,fps,frame_start)
        if fps!=0:
            try:
                sleep(1/fps - (time.time()-frame_start))
            except:
                print("Remain time is minus")
        return up_scale_image,result

    def mp_dic2image_frame(self,global_out_image,current_pose_dic,img_number,user_id,mode,scale,fps):
        print(current_pose_dic)
        frame_start=time.time()
        current_pose=self.get_pose_dic(current_pose_dic)
        out_image=self._mp_inference_img(global_out_image, current_pose,img_number,user_id,out_typ="cv2")
        up_scale_image ,result= self._mp_get_upscale(global_out_image,out_image,mode,scale,fps,frame_start)
        if fps!=0:
            try:
                sleep(1/fps - (time.time()-frame_start))
            except:
                print("Remain time is minus and sleep=0")
        return up_scale_image,result

    def _mp_inference_img(self,global_out_image, current_pose,img_number,user_id,out_typ="cv2"):
        if self.queue_tkh_pose.empty()==True:
            send_data=[current_pose , img_number , user_id,out_typ]
            self.queue_tkh_pose.put(send_data)
        if self.queue_tkh_image.empty()==False:
            get_out_image = self.queue_tkh_image.get() # queue_in_imageからデータを取得
            self.previous_image=get_out_image
        else:
            print("-----Talking Head Skip")
            get_out_image=self.previous_image #<<<<global_out_imageは拡大後のイメージなので、前回のTKHの生成イメージが必要
              
        return get_out_image

    def _mp_get_upscale(self,global_out_image,out_image,mode,scale,fps,frame_start):
        result=True
        if self.queue_in_image.empty()==True:
            send_data=[out_image , mode , scale]
            self.queue_in_image.put(send_data)
        else:
            print("++@@@+++ Upscale skip 1") #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #print(" ++@@@+++ queue_out_image=",self.queue_out_image.qsize()) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   
        if self.queue_out_image.empty()==False:
            
            global_out_image = self.queue_out_image.get() # queue_in_imageからデータを取得
        else:
            print("++@@@+++ Upscale skip 2") #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            result=False
        #else:
        #    result=False
        #    print("++@@@+++ Upscale skip")
        #try:
        #    sleep(1/fps - (time.time()-frame_start))
        #except:
        #    print("+++++ Upscale Remain time is minus")
        return global_out_image,result

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
        #print("++++++ dic=",dic)
        current_dic = pickle.dumps(dic, 5)
        files = {"pose":("pos.dat",current_dic, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/get_pose_dic/", files=files) #リクエスト
        if response.status_code == 200:
            pose_data = response.content
            pose =(pickle.loads(pose_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return pose   

    def load_img(self,input_img,user_id):
        print("load_img")
        images_data = pickle.dumps(input_img, 5) 
        files = {"image": ("img.dat",  images_data, "application/octet-stream")}
        data = {"user_id": user_id}
        response = requests.post(self.url+"/load_img/", files=files, data=data) #リクエスト送信
        if response.status_code == 200:
            response_data = response.json()
            print("response_data =",response_data)
            img_number=response_data["img_number"]
        else:
            img_number=-1
        return img_number

    def inference(self,input_img,current_pose,out="pil"):#基本イメージ生成、イメージは毎回ロード
        start_time=time.time()
        images_data = pickle.dumps(input_img, 5)
        current_pose2 = pickle.dumps(current_pose, 5)
        files = {"image": ("img.dat",images_data, "application/octet-stream"),
                 "pose":("pos.dat",current_pose2, "application/octet-stream"),
                 "out":("out.dat", out, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/inference_org/", files=files) #リクエスト
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image
    
    def inference_pos(self,packed_pose,img_number,user_id,out="pil"):#イメージは事前ロード
        packed_pose = pickle.dumps(packed_pose, 5)
        files={"pose":("pos.dat",packed_pose, "application/octet-stream"),}
              # "img_number":img_number,
              # "user_id": user_id,}#listで渡すとエラーになる
        data = {"user_id": user_id,"img_number":img_number,"out":out}
        response = requests.post(self.url+"/inference_pos/", files=files, data=data) #リクエスト送信
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image

    def inference_dic(self,current_dic,img_number,user_id,out="pil"):#イメージは事前ロード
        data = {"img_number":img_number,"user_id": user_id,"out":out}
        current_dic2 = pickle.dumps(current_dic, 5)
        files={"pose":("pos.dat",current_dic2, "application/octet-stream")}#listで渡すとエラーになる
        response = requests.post(self.url+"/inference_dic/", data=data,files=files) #リクエスト送信
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image
        
    def inference_img(self,current_pose,img_number,user_id,out="pil"):#イメージ事前ロード用生成 イメージは事前ロード
        data = {"current_pose":current_pose,"img_number":img_number,"user_id": user_id,"out":out}
        response = requests.post(self.url+"/inference_img/", data=data) #リクエスト送信
        if response.status_code == 200:
            image_data = response.content
            image =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
            result = response.status_code
        return result, image


