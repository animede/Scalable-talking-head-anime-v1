import numpy as np
import cv2
from PIL import Image
import argparse
from time import sleep
import time
import os
import signal
from poser_generater_v1_3 import TalkingHeadAnimefaceGenerater
    
#サンプル 1　inference_dic() 　poseはDICT形式で直接サーバを呼ぶ　イメージは事前ロード　  DICT形式で必要な部分のみ選んで連続変化させる
def auto_pose_1(Thi,test,input_image,user_id,img_number,mode,scale,fps):
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
        #TalkingHeadAnimefaceGenerater 定義と初期化
        Tkg=TalkingHeadAnimefaceGenerater(Thi,img_number,user_id,mode,scale,fps)
        #ポーズデータ生成プロセスのスタート
        pid_mp_gen=Tkg.start_mp_generater_process()
        #pose_dic=pose_dic_org #Pose 初期値
        current_pose_dic=pose_dic_org #Pose 初期化
        #単純なコマ割り数
        div_count=300
        #生成フレーム速度など初期化
        move_time=div_count/2*(1/fps)
        #Head pose 動作有効化
        Tkg.pose_head(0.0, 3.0, 3.0, move_time, current_pose_dic)#head_x,head_y,neck,time,current_pose_dic
        #Head body 動作有効化
        Tkg.pose_body(3.0, 3.0, 3.0, move_time, current_pose_dic)#body_y, body_z, breathing,time,current_pose_dic
        #auto_eye_blink開始
        pid_mp_eye=Tkg.mp_auto_eye_blink_start(1,2)
        #mouth_list  定義　以下の例ではあ、い、う、え、お、あ、の順で口を動かす
        mouth_list=["aaa","iii","uuu","eee","ooo","aaa"]
        mouth_pointer=0
        if test=="test1":
            #1回目の動きループ　口の動き
            for i in range(int(div_count/2)):
                start_time=time.time()
                # mouthe pose
                if (i==50 or i==60 or i==70 or i==80 or i==100):
                    mouth_menue = mouth_list[mouth_pointer]
                    Tkg.pose_mouth(mouth_menue, 1.0, 0.1, current_pose_dic)
                    mouth_pointer +=1
                if (i==130):
                    Tkg.pose_mouth("aaa", 0.0, 0.1, current_pose_dic)
                # mabataki pose
                if (i==20 or i==50):
                    Tkg.pose_wink("b", 0.15,current_pose_dic)#l_r,time
                # wink pose
                if (i==10 or i==30):
                    Tkg.pose_wink("l", 0.2,current_pose_dic)#l_r,time
                if (i==65):
                    Tkg.pose_wink("r", 0.2,current_pose_dic)#l_r,time
                # iris pose
                if (i==5 or i==75):
                    Tkg.pose_iris(1.0, 0.0, 0.1,current_pose_dic)#small,rotation,time
                if (i==25 or i==85):
                    Tkg.pose_iris(0.0, 0.0, 0.15,current_pose_dic)#small,rotation,time
                if (i==140):
                    Tkg.pose_face("happy", 0.0, "happy_wink", 0.0, 0.5,current_pose_dic)#happy :eyebrow_menue, eyebrow, eye_menue, eye, time,current_pose_dic
                #画像の取得
                result_out_image, current_pose_dic = Tkg.get_image()
                
                cv2.imshow("pose image",result_out_image)
                cv2.waitKey(1)
                #cv2.imwrite("image1/image"+str(i+1000)+".jpg",result_out_image)
                print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
                if (1/fps - (time.time()-start_time))>0:
                    sleep(1/fps - (time.time()-start_time))
                else:
                    print("Remain time is minus")
                print("Genaration time=",(time.time()-start_time)*1000,"mS")
            #Head pose　動作開始
            Tkg.pose_head(0.0, -3.0, 0.0, move_time, current_pose_dic)#head_x,head_y,neck,time,current_pose_dic
            # body pose 動作開始
            Tkg.pose_body(-6.0, -3.0, -3.0, move_time, current_pose_dic)#body_y, body_z, breathing,time,current_pose_dic
        elif test=="test2":
            #２回目の動きループ　感情表現１
            for i in range(int(div_count/2)):
                start_time=time.time()
                #Emotion 指定　→ "happy" #喜 "angry" #怒 "sorrow" #哀 "relaxed" #楽 "smile" #微笑む "laugh" #笑う "surprised" #驚く
                if i==20:
                    Tkg.pose_emotion("happy",0.5, current_pose_dic)
                if i==60:
                    Tkg.pose_emotion("angry", 0.5, current_pose_dic)
                if i==100:
                    Tkg.pose_emotion("sorrow", 0.5, current_pose_dic)
                if i==140:
                    Tkg.pose_emotion("relaxed", 0.5, current_pose_dic)
                #画像の取得
                result_out_image, current_pose_dic = Tkg.get_image()
                
                cv2.imshow("pose image",result_out_image)
                cv2.waitKey(1)
                #cv2.imwrite("image1/image2"+str(i+2000)+".jpg",result_out_image)
                print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
                if (1/fps - (time.time()-start_time))>0:
                    sleep(1/fps - (time.time()-start_time))
                else:
                    print("Remain time is minus")
                print("Genaration time=",(time.time()-start_time)*1000,"mS")

            #Head pose　動作開始
            Tkg.pose_head(0.0, 0.0, 0.0, move_time, current_pose_dic)#head_x,head_y,neck,time,current_pose_dic
            # body pose 動作開始
            Tkg.pose_body(0.0, 0.0, 0.0, move_time, current_pose_dic)#body_y, body_z, breathing,time,current_pose_dic
        elif test=="test3":
            #３回目の動きループ　感情表現２
            for i in range(int(div_count/2)):
                start_time=time.time()
                #Emotion 指定　→ "happy" #喜 "angry" #怒 "sorrow" #哀 "relaxed" #楽 "smile" #微笑む "laugh" #笑う "surprised" #驚く
                if i==20:
                    Tkg.pose_emotion("laugh", 0.5, current_pose_dic)
                if i==60:
                    Tkg.pose_emotion("surprised", 0.2, current_pose_dic)   
                if i==800:
                    Tkg.pose_emotion("smile", 0.5, current_pose_dic)
                if i==100:
                    Tkg.pose_face("happy", 0.0, "happy_wink", 0.0, 0.5,current_pose_dic)#happy :eyebrow_menue, eyebrow, eye_menue, eye, time,current_pose_dic
                    Tkg.pose_mouth("aaa", 0.0, 0.5, current_pose_dic)
                #画像の取得
                result_out_image, current_pose_dic = Tkg.get_image()
                
                cv2.imshow("pose image",result_out_image)
                cv2.waitKey(1)
                #cv2.imwrite("image1/image3"+str(i+3000)+".jpg",result_out_image)
                print("1/fps - (time.time()-start_time)=",1/fps - (time.time()-start_time))
                if (1/fps - (time.time()-start_time))>0:
                    sleep(1/fps - (time.time()-start_time))
                else:
                    print("Remain time is minus")
                print("Genaration time=",(time.time()-start_time)*1000,"mS")
            
        #終了時に最後の画像を１０秒表示する
        cv2.imshow("pose image",result_out_image)
        cv2.waitKey(5)
        cv2.waitKey(1000)
        cv2.destroyWindow("pose image")
        #サブプロセスの終了
        os.kill(pid_mp_gen, signal.SIGKILL)
        os.kill(pid_mp_eye, signal.SIGKILL)
        
        print("end of test")

