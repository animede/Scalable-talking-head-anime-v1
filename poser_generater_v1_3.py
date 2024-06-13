import numpy as np
import cv2
from PIL import Image
import argparse
from time import sleep
import time
import random
import multiprocessing
from multiprocessing import Process, Value, Manager
from poser_client_tkhmp_upmp_v1_3_class import TalkingHeadAnimefaceInterface
from tkh_up_scale import upscale

#PIL形式の画像を動画として表示 TEST用
def image_show(imge):
    imge = np.array(imge)
    imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Loaded image",imge)
    cv2.waitKey(1)

#-----Dict形式
#{"eyebrow":{"menue":"happy","left":0.0,"right":0.0}, #menue: "troubled", "angry", "lowered", "raised", "happy", "serious"
#  "eye":{"menue":"wink","left":0.0,"right":0.0},     #menue: "wink", "happy_wink", "surprised", "relaxed", "unimpressed", "raised_lower_eyelid"
#  "iris_small":{"left":0.0,"right":0.0},
#  "iris_rotation":{"x":0.0,"y":0.0},
#  "mouth":{"menue":"aaa","val":0.0},                 #menue: "aaa", "iii", "uuu", "eee", "ooo", "delta", "lowered_corner", "raised_corner", "smirk"
#  "head":{"x":0.0,"y":0.0},
#  "neck":0.0,
#  "body":{"y":0.0,"z":0.0},
#  "breathing":0.0,
#  }

class TalkingHeadAnimefaceGenerater():
        
    def __init__(self,Thi,img_number,user_id,mode,scale,fps):
        self.img_number=img_number
        self.user_id=user_id
        self.mode=mode
        self.scale=scale
        self.fps=fps
        self.pose_dic_org={"eyebrow":{"menue":"happy","left":0.0,"right":0.0},
                  "eye":{"menue":"wink","left":0.0,"right":0.0},
                  "iris_small":{"left":0.0,"right":0.0},
                  "iris_rotation":{"x":0.0,"y":0.0},
                  "mouth":{"menue":"aaa","val":0.0},
                  "head":{"x":0.0,"y":0.0},
                  "neck":0.0,
                  "body":{"y":0.0,"z":0.0},
                  "breathing":0.0,
                  }
        self.current_poce_dic= self.pose_dic_org
        
        #Thiの初期化
        self.Thi=Thi
        self.q_in_wink = None  #ウインク、両目の瞬き
        self.q_in_iris = None  #
        self.q_in_face = None  #顔の感情
        self.q_in_mouse = None #リップシンク
        self.q_in_head = None  #頭を動かす
        self.q_in_body = None  #体を動かす
        self.q_in_pose = None  #全Pose_Dicで動かす
        self.queue_out_image = None  #出力画像

        self.out_image = np.zeros((512, 512, 3), dtype=np.uint8)#upscaleの最初で呼び出す初期画像、その後返送用のイメージバッファになる
        self.global_out_image=self.out_image#_mp_generataerでアップスケース画像を保存
    #start process
    def start_mp_generater_process(self):
        self.q_in_wink = multiprocessing.Queue()  # 入力Queueを作成  ウインク
        self.q_in_iris = multiprocessing.Queue()  # 入力Queueを作成  瞳
        self.q_in_face = multiprocessing.Queue()  # 入力Queueを作成  顔の感情
        self.q_in_mouth= multiprocessing.Queue()  # 入力Queueを作成 リップシンク
        self.q_in_head = multiprocessing.Queue()  # 入力Queueを作成  頭を動かす
        self.q_in_body = multiprocessing.Queue()  # 入力Queueを作成  体を動かす
        self.q_in_pose = multiprocessing.Queue()  # 入力Queueを作成  #全Pose_Dicで動かす
        self.queue_out_image = multiprocessing.Queue()  # 出力Queueを作成

        self.mp_generater_proc = multiprocessing.Process(target=self._mp_generataer,
                                                         args=(self.Thi,
                                                               self.pose_dic_org,
                                                               self.global_out_image,
                                                               self.queue_out_image,
                                                               self.img_number,
                                                               self.user_id,
                                                               self.mode,
                                                               self.scale,
                                                               self.fps,
                                                               self.q_in_wink,
                                                               self.q_in_iris,
                                                               self.q_in_face,
                                                               self.q_in_mouth,
                                                               self.q_in_head,
                                                               self.q_in_body,
                                                               self.q_in_pose,)) 
        self.mp_generater_proc.start() #process開始
        #self.mp_generater_proc.join()

        
        pid=self.mp_generater_proc.pid #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        return pid                 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    #_mp_generatae(): 部位別POSEリクエストを確認してリクエストがあれば各部のPOSEを反映する
    def _mp_generataer(self, Thi, pose_data_dic , global_out_image , out_image , img_number , user_id , mode , scale,fps,
                                                               q_in_wink,
                                                               q_in_iris,
                                                               q_in_face,
                                                               q_in_mouth,
                                                               q_in_head,
                                                               q_in_body,
                                                               q_in_pose,):
        print("+++++ Start mp_generataer process")
        pose_request=False
        wink_request=False
        iris_request=False
        face_request=False
        mouth_request=False
        head_request=False
        body_request=False
        #lOOP開始　停止はプロセスのTERMINATE
        while True:
            start_time=time.time()
            #ポーズリクエストQueueの監視と解析
            if self.q_in_wink.empty()==False:#***
                wink_data=self.q_in_wink.get()
                wink_step_right=wink_data[1]
                wink_step_left =wink_data[2]
                wink_step_count=wink_data[0]
                wink_step_count_p=wink_step_count
                wink_step_count_n=wink_step_count
                wink_request=True
                pose_request=True
                print("+++++ wink_data=",wink_data)
            elif self.q_in_iris.empty()==False:#***
                iris_data=self.q_in_iris.get()
                iris_small_step_left=iris_data[1]
                iris_small_step_right =iris_data[2]
                iris_rotation_step_x =iris_data[3]
                iris_rotation_step_y =iris_data[4]
                iris_step_count=iris_data[0]                
                iris_request=True
                pose_request=True
                print("+++++ iris_data=",iris_data)
            elif self.q_in_face.empty()==False:#***
                face_data=self.q_in_face.get()
                eyebrow_step_right=face_data[1]
                eyebrow_step_left =face_data[2]
                eyebrow_menue     =face_data[3]
                face_eye_step_right=face_data[4]
                face_eye_step_left =face_data[5]
                face_eye_menue     =face_data[6]
                face_step_count     =face_data[0]
                face_request=True
                pose_request=True
                print("+++++ face_data=",face_data)
            elif self.q_in_mouth.empty()==False:#***
                mouth_data=self.q_in_mouth.get()
                mouth_step=mouth_data[2]
                mouth_menue=mouth_data[1]
                mouth_step_count=mouth_data[0]
                mouth_request=True
                pose_request=True
                print("+++++ mouth_data=",mouth_data)
            elif self.q_in_head.empty()==False:#***
                head_data=self.q_in_head.get()
                head_step_x=head_data[1]
                head_step_y=head_data[2]
                neck_step=head_data[3]
                head_step_count=head_data[0]
                head_request=True
                pose_request=True
                print("+++++ head_data",head_data)
            elif self.q_in_body.empty()==False:#***
                body_data=self.q_in_body.get()
                body_step_y=body_data[1]
                body_step_z=body_data[2]
                breath_step=body_data[3]
                body_step_count=body_data[0]
                body_request=True
                pose_request=True
                print("+++++ body_data",body_data)
            elif self.q_in_pose.empty()==False:# (Pose Dic全体が送られて来る)=>改め、生成画像を取得する役目に
                pose_data=self.q_in_pose.get()
                pose_request=True #他のリクエストがなければ画像と最新pose_dataを送信する
                
            #リクエストがあればポーズデータをStep数とStep値で更新していく。if文なので各々の変化も同時に更新できる
            if pose_request:
                if wink_request:#*** wink(eye)
                    pose_data_dic["eye"]["menue"]="wink"
                    if wink_step_count_p>0:
                        pose_data_dic["eye"]["right"] +=wink_step_right
                        pose_data_dic["eye"]["left"]  +=wink_step_left
                        wink_step_count_p -=1
                    elif wink_step_count_n>0:
                        pose_data_dic["eye"]["right"]-=wink_step_right
                        pose_data_dic["eye"]["left"] -=wink_step_left
                        wink_step_count_n -=1
                    else:
                        wink_request=False
                        
                if iris_request:#*** iris
                    if iris_step_count>0:
                        pose_data_dic["iris_small"]["right"] +=iris_small_step_right
                        pose_data_dic["iris_small"]["left"]  +=iris_small_step_left
                        pose_data_dic["iris_rotation"]["x"]  +=iris_rotation_step_x
                        pose_data_dic["iris_rotation"]["y"]  +=iris_rotation_step_y
                        iris_step_count -=1
                    else:
                        iris_request=False

                if face_request:#*** face (eyebrow_menue + eye)
                    pose_data_dic["eyebrow"]["menue"] =eyebrow_menue
                    pose_data_dic["eye"]["menue"]=face_eye_menue
                    if face_step_count>0:
                        pose_data_dic["eyebrow"]["right"]+=eyebrow_step_right
                        pose_data_dic["eyebrow"]["left"] +=eyebrow_step_left
                        pose_data_dic["eye"]["right"]    +=face_eye_step_right
                        pose_data_dic["eye"]["left"]     +=face_eye_step_left
                        face_step_count -=1
                    else: 
                        face_request=False
                        
                if mouth_request:#*** mouth
                    pose_data_dic["mouth"]["menue"]=mouth_menue
                    if mouth_step_count>0:
                        pose_data_dic["mouth"]["val"] +=mouth_step
                        mouth_step_count -=1
                    else:
                        mouth_request=False
                        
                if head_request:#*** head (head + neck)
                    if head_step_count>0:
                        pose_data_dic["head"]["x"] +=head_step_x
                        pose_data_dic["head"]["y"] +=head_step_y
                        pose_data_dic["neck"] +=neck_step
                        head_step_count -=1
                    else:
                        head_request=False

                if body_request:#*** body (body + breath
                    if body_step_count>0:
                        pose_data_dic["body"]["y"] +=body_step_y
                        pose_data_dic["body"]["z"] +=body_step_z
                        pose_data_dic["breathing"] +=breath_step
                        body_step_count -=1
                    else:
                        body_request=False
                       
                #新しいポーズデータができたので画像の生成を依頼する 
                out_image,result=Thi.mp_dic2image_frame(global_out_image, pose_data_dic ,img_number, user_id, mode, scale, 0)
                
                #生成が間に合わなかった場合は画像をqueueに送信しない
                if result:
                    if self.queue_out_image.empty():
                        send_data=[out_image , pose_data_dic]
                        self.queue_out_image.put(send_data)
            print("mp_generataer : remain time to frame rate=",1/self.fps - (time.time()-start_time))
            if (1/self.fps - (time.time()-start_time))>0:
                sleep(1/self.fps - (time.time()-start_time))
            else:
                print("mp_generatae Remain time is minus")
            pose_request=False
            sleep(0.005)
            
    #mp_generater_processプロセス停止関数--terminate
    def mp_generater_process_terminate(self):
        while not self.queue_out_image.empty():
            self.queue_out_image.get_nowait()
        while not self.q_in_pose.empty():
            self.q_in_pose.get_nowait()
        self.mp_generater_proc.terminate()#サブプロセスの終了
        print("mp_generataer process terminated")

    def mp_auto_eye_blink_start(self,start,end):
        self.mp_auto_eye_blink_proc = multiprocessing.Process(target=self._mp_auto_eye_blink,args=(start,end,self.pose_dic_org))
        self.mp_auto_eye_blink_proc.start() #process開始
        pid=self.mp_auto_eye_blink_proc.pid
        return pid
    
    def _mp_auto_eye_blink(self,start,end,pose_data_dic):
        print("++++++ Start mp_auto_eye_blink")
        while True:
            sleep(random.randint(start, end))
            print(pose_data_dic)
            self.pose_wink("b", 0.1, pose_data_dic)
            
    def mp_auto_eye_blink_teminate(self):
         self.mp_auto_eye_blink_proc.terminate()

            
        
    # wink(瞬きも)ポーズリクエスト
    def pose_wink(self,l_r, time,current_pose_dic):
        if self.q_in_wink.empty():
            step_count=int((time/(1/self.fps))/2)
            if l_r=="l":
                if self.q_in_wink.empty():
                    eye_left=(1-current_pose_dic["eye"]["left"])/step_count
                    send_data=[step_count,0.0,eye_left]
                    self.q_in_wink.put(send_data)
            elif l_r=="r":
                if self.q_in_wink.empty():
                    eye_right=(1-current_pose_dic["eye"]["right"])/step_count
                    send_data=[step_count,eye_right,0.0]
                    self.q_in_wink.put(send_data)
            elif l_r=="b":
                if self.q_in_wink.empty():
                    eye_left=(1-current_pose_dic["eye"]["left"])/step_count
                    eye_right=(1-current_pose_dic["eye"]["right"])/step_count
                    send_data=[step_count,eye_right,eye_left]
                    self.q_in_wink.put(send_data)
            print("===============pose_wink: l_r=",l_r," send_data=",send_data)
            
    # 瞳ポーズリクエスト(左右同時に簡略化）　通常はこちら
    def pose_iris(self, iris_small, iris_rotation, time,current_pose_dic):
        self.pose_iris_sp(iris_small, iris_small, iris_rotation, iris_rotation,time,current_pose_dic)
        
    # 瞳ポーズリクエスト（左右個別）
    def pose_iris_sp(self, iris_l, iris_r, iris_x,iris_y,time,current_pose_dic):
        if self.q_in_iris.empty():
            step_count=int(time/(1/self.fps))
            iris_l_step = (iris_l-current_pose_dic["iris_small"]["left"])/step_count
            iris_r_step = (iris_r-current_pose_dic["iris_small"]["right"])/step_count
            iris_x_step = (iris_x-current_pose_dic["iris_rotation"]["x"])/step_count
            iris_y_step = (iris_y-current_pose_dic["iris_rotation"]["y"])/step_count
            send_data=[step_count,iris_l_step ,iris_r_step ,iris_x_step, iris_y_step]
            self.q_in_iris.put(send_data)
            print("===============pose_iris:  send_data=",send_data)
            
    # 顔ポーズリクエスト(左右同時に簡略化）　通常はこちら
    def pose_face(self, eyebrow_menue, eyebrow, eye_menue, eye, time,current_pose_dic):
        self.pose_face_sp(eyebrow_menue, eyebrow, eyebrow,  eye_menue, eye, eye, time,current_pose_dic)
        
    # 顔ポーズリクエスト（左右個別）
    def pose_face_sp(self, eyebrow_menue, eyebrow_l, eyebrow_r,  eye_menue, eye_r, eye_l, time,current_pose_dic):
        if self.q_in_face.empty():
            step_count=int(time/(1/self.fps))
            eyebrow_l_step = (eyebrow_l - current_pose_dic["eyebrow"]["left"])/step_count
            eyebrow_r_step = (eyebrow_r - current_pose_dic["eyebrow"]["right"])/step_count
            eye_l_step = (eye_l - current_pose_dic["eye"]["left"])/step_count
            eye_r_step = (eye_r - current_pose_dic["eye"]["right"])/step_count
            send_data=[step_count,eyebrow_l_step ,eyebrow_r_step ,eyebrow_menue, eye_r_step, eye_l_step, eye_menue]
            self.q_in_face.put(send_data)
            print("===============pose_face:  send_data=",send_data)
            
    # 口形状リクエスト
    def pose_mouth(self,mouth_menue, mouth_val, time, current_pose_dic):
        if self.q_in_mouth.empty():
            step_count=int(time/(1/self.fps))
            mouth_val_step = (mouth_val-current_pose_dic["mouth"]["val"])/step_count
            send_data=[step_count, mouth_menue ,mouth_val_step]
            self.q_in_mouth.put(send_data)
            print("===============pose_mouth:  send_data=",send_data)
            
    # 頭ポーズリクエスト
    def pose_head(self,head_x,head_y,neck,time,current_pose_dic):
        if self.q_in_head.empty():
            step_count=int(time/(1/self.fps))
            head_x_step = (head_x-current_pose_dic["head"]["x"])/step_count
            head_y_step = (head_y-current_pose_dic["head"]["y"])/step_count
            neck_step = (neck-current_pose_dic["neck"])/step_count
            send_data=[step_count,head_x_step ,head_y_step ,neck_step]
            self.q_in_head.put(send_data)
            print("===============pose_head:  send_data=",send_data)
            
    # 体ポーズリクエスト
    def pose_body(self, body_y, body_z, breathing, time, current_pose_dic):
        if self.q_in_body.empty():
            step_count=int(time/(1/self.fps))
            body_y_step = (body_y-current_pose_dic["body"]["y"])/step_count
            body_z_step = (body_z-current_pose_dic["body"]["z"])/step_count
            breathi_step = (breathing-current_pose_dic["breathing"])/step_count
            send_data=[step_count,body_y_step ,body_z_step ,breathi_step]
            self.q_in_body.put(send_data)
            print("===============pose_body:  send_data=",send_data)
            
    # 感情ポーズリクエスト  "happy":#喜 "angry":#怒 "sorrow":#哀 relaxed":#楽 "smile":#微笑む "laugh":#笑う "surprised":#驚く
    def pose_emotion(self, menue, time, current_pose_dic):
        if menue=="happy":#喜
            self.pose_face("happy", 1.0, "happy_wink", 1.0, time, current_pose_dic)
            self.pose_mouth("iii", 1.0, time, current_pose_dic)
            
        elif menue=="angry":#怒
            self.pose_face("angry", 1.0, "raised_lower_eyelid", 1.0, time, current_pose_dic)
            self.pose_mouth("uuu", 1.0, time, current_pose_dic)
            
        elif menue=="sorrow":#哀
            self.pose_face("troubled", 1.0, "unimpressed", 1.0, time, current_pose_dic)
            self.pose_mouth("ooo", 1.0, time, current_pose_dic)
            
        elif menue=="relaxed":#楽
            self.pose_face("happy", 1.0, "relaxed", 1.0, time, current_pose_dic)
            self.pose_mouth("aaa", 0.0, time, current_pose_dic)
            
        elif menue=="smile":#微笑む
            self.pose_face("happy", 1.0, "relaxed", 1.0, time, current_pose_dic)
            self.pose_mouth("aaa", 0.0, time, current_pose_dic)
            
        elif menue=="laugh":#笑う
            self.pose_face("happy", 1.0, "wink", 0.0, time, current_pose_dic)
            self.pose_mouth("aaa", 1.0, time, current_pose_dic)
            
        elif menue=="surprised":#驚く
            self.pose_face("lowered", 1.0, "surprised", 1.0, time, current_pose_dic)
            self.pose_mouth("aaa", 1.0, time, current_pose_dic)
            
        else:
            print("Emotion Error")
            
    # 画像の取得
    def get_image(self):#as Get image
        if self.q_in_pose.empty():
            self.q_in_pose.put("get_image")   #送るデータは何でも良い 
        if self.queue_out_image.empty()==False:
            recive_data = self.queue_out_image.get()
            self.out_image = recive_data[0]   #_mp_generataerから送られてきた生成画像
            self.current_poce_dic = recive_data[1]
        return self.out_image ,self.current_poce_dic
    
    #全プロセス停止
    def mp_all_proc_terminate(self):
        self.mp_generater_process_terminate()
        self.mp_auto_eye_blink_teminate()
