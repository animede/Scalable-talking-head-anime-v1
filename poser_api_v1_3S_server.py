import time
import pickle
from  poser_api_v1_3_class import TalkingHeadAnimeface

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from poser_image_2_template_class import Image2form
import subprocess
import signal
import sys
global process_u

app = FastAPI()

#process_u = subprocess.Popen(['gnome-terminal', '--', 'python3', 'realesr_api_server.py']) #Terminalを開いてUPSCALEサーバを動かす
process_u = subprocess.Popen(['python3', 'realesr_api_server.py']) #Terminalを開かずにUPSCALEサーバを動かす

def signal_handler(signal, frame):
    global process_u
    print("Ctrl-C pressed: Exiting...")
    process_u.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

Tkh=TalkingHeadAnimeface()
up_url="http://0.0.0.0:8008/resr_upscal/"
bkl_url="http://0.0.0.0:8007/del_bkg/"
I2f=Image2form(up_url)

# 新しいターミナルを開き、特定のPythonスクリプトを実行する
#process_u = subprocess.run('gnome-terminal -- python3 realesr_api_server.py', shell=True)

#process_u = subprocess.Popen(['gnome-terminal', '--', 'python3', 'realesr_api_server.py'])

@app.post("/image_2_form/")
def image_2_form(image: UploadFile = File(...), image_mode:str = Form("pil")):
    file_contents = image.file.read()
    img_data =(pickle.loads(file_contents))#元の形式にpickle.loadsで復元

    result , form_img = I2f.image_data_form(img_data,image_mode)
    if result != True:
         form_img ="Error"        
    frame_data = pickle.dumps(form_img , 5)
    return Response(content=frame_data, media_type="application/octet-stream")

@app.post("/get_init_dic/")
def get_init_dic():
    pose_dic_org=Tkh.get_init_dic()
    org_dic = pickle.dumps(pose_dic_org,5)
    return Response(content= org_dic, media_type="application/octet-stream")
    
@app.post("/load_img/")
def load_img(image: UploadFile = File(...),user_id:int= Form(...)):
    image_data = image.file.read()
    image_data =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
    image_data = image_data.convert("RGBA")
    img_number=Tkh.load_img(image_data,user_id)
    result="OK"
    print("img_number=",img_number)
    return {'message':result,"img_number":img_number}

@app.post("/inference_org/")
def inference_org(image:UploadFile = File(...),pose:UploadFile = File(...),out:UploadFile = File(...)): #基本イメージ生成、イメージは毎回ロード
    image_data = image.file.read()
    current_pose = pose.file.read()
    out =(out.file.read()).decode('utf-8') 
    image_data =(pickle.loads(image_data))#元の形式にpickle.loadsで復元
    input_image = image_data.convert("RGBA")
    current_pose=(pickle.loads(current_pose))
    out_image=Tkh.inference(input_image,current_pose,out)
    #−−−−−生成画像を返信
    images_data = pickle.dumps(out_image, 5)  # tx_dataはpklデータ
    return Response(content= images_data, media_type="application/octet-stream")

@app.post("/inference_pos/")
def inference_pos(pose:UploadFile = File(...),img_number:int= Form(...),user_id:int= Form(...),out:str= Form(...)):
    start_time=time.time()
    packed_pose = pose.file.read()
    packed_pose=(pickle.loads(packed_pose))
    #print(packed_pose)
    out_image=Tkh.inference_pos(packed_pose,img_number,user_id,out)
    #−−−−−生成画像を返信
    images_data = pickle.dumps(out_image, 5)  # tx_dataはpklデータ
    print("Tkh genaration time=",(time.time()-start_time)*1000,"mS")
    return Response(content= images_data, media_type="application/octet-stream")

@app.post("/inference_dic/")
def inference_dic(pose:UploadFile = File(...),img_number:int= Form(...),user_id:int= Form(...),out:str= Form(...)):
    start_time=time.time()
    current_dic = pose.file.read()
    current_pose_dic=(pickle.loads(current_dic))
    out_image=Tkh.inference_dic(current_pose_dic,img_number,user_id,out)
    #−−−−−生成画像を返信
    images_data = pickle.dumps(out_image, 5)  # tx_dataはpklデータ
    print("Tkh genaration time=",(time.time()-start_time)*1000,"mS")
    return Response(content= images_data, media_type="application/octet-stream")

@app.post("/inference_img/")
def inference_img(current_pose:list= Form(...),img_number:int= Form(...),user_id:int= Form(...),out:str= Form(...)):
    start_time=time.time()
    current_pose = [float(item) for item in current_pose]
    out_image=Tkh.inference_img(current_pose,img_number,user_id,out)
    #−−−−−生成画像を返信
    images_data = pickle.dumps(out_image, 5)  # tx_dataはpklデータ
    print("Tkh genaration time=",(time.time()-start_time)*1000,"mS")
    return Response(content= images_data, media_type="application/octet-stream")

@app.post("/get_pose/")
def get_pose(pose:UploadFile = File(...)):
    pkl_pack= pose.file.read()
    pose_pack=(pickle.loads(pkl_pack))
    pose=Tkh.get_pose(pose_pack)
    pose_data = pickle.dumps(pose, 5)  # tx_dataはpklデータ
    return Response(content= pose_data, media_type="application/octet-stream")

@app.post("/get_pose_dic/")
def get_pose_dic(pose:UploadFile = File(...)):
    pose= pose.file.read()
    pose_dic=(pickle.loads(pose))
    pose=Tkh.get_pose_dic(pose_dic)
    pose_data = pickle.dumps(pose, 5)  # tx_dataはpklデータ
    return Response(content= pose_data, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
 
