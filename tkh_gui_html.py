from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import base64
import numpy as np
from PIL import Image
import io
import cv2
from time import sleep
import signal
import sys
import json
from poser_client_tkhmp_upmp_v1_3_class import TalkingHeadAnimefaceInterface

global result_out_image
global img_mode
global img_number
global current_pose_dic

app = FastAPI()
# 静的ファイルを提供するためのディレクトリをマウント
tkh_url='http://0.0.0.0:8001'
esr_url='http://0.0.0.0:8008/resr_upscal/'
app.mount("/static", StaticFiles(directory="static"), name="static")
Thi = TalkingHeadAnimefaceInterface(tkh_url)
pose_dic_org = Thi.get_init_dic()
pose_dic=pose_dic_org.copy() #Pose 初期値
current_pose_dic=pose_dic.copy()
img_number = 0
user_id=0
#アップスケールとtkhプロセスの開始
Thi.create_mp_upscale(esr_url)
Thi.create_mp_tkh()

def signal_handler(signal, frame):
    print("Ctrl-C pressed: Exiting...")
    Thi.up_scale_proc_terminate()
    Thi.tkh_proc_terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)  

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open('static/index.html', 'r') as f:
        return f.read()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    global result_out_image
    global img_number
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents))
    result_image=Thi.image_2_form(input_image, "pil")
    cv2_image = np.array(result_image, dtype=np.uint8)
    result_out_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2BGRA)
    img_number = Thi.load_img(result_image, user_id=0)

    buffered = io.BytesIO()
    result_image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {"original": base64.b64encode(contents).decode('utf-8'), "processed": encoded_string,"img_number":img_number}

@app.post("/generate_image/")
def generate_image(mode: str = Form(...), scale: int = Form(...), fps: int = Form(...)):
    global result_out_image
    global img_mode
    global img_number
    global current_pose_dic

    try:
        cv2_image = np.array(result_image, dtype=np.uint8)
        result_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    except:
        print("image=cv2")
    if len(mode)>9:  # <= modeがクロップ用の位置情報リストで来た場合。すべての要素が1桁の場合やmodeの文字が10を超えるとだめです
        img_mode = json.loads(mode)
    else:
        img_mode=mode
    user_id=0
    result_out_image,_ = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps=0)
    sleep(0.1)
    result_out_image,_ = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps=0)
    sleep(0.1)
    result_out_image,_ = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps=0)
    sleep(0.1)
    # 処理された画像をエンコードして返送
    cv2_image = cv2.cvtColor(result_out_image, cv2.COLOR_BGRA2RGBA)
    out_image = Image.fromarray(cv2_image)  
    buffered = io.BytesIO()
    out_image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {"processed": encoded_string}

class AdjustmentData(BaseModel):
    eyebrow_type: str
    eye_type: str
    mouth_type: str
    adjustment_type: str
    adjustment_value: float  # floatとして定義されていることを確認
    scale: str
    fps: str
    useOpenCV:str
@app.post("/update_adjustment/")
async def update_adjustment(data: AdjustmentData):
    global result_out_image
    global img_mode
    global img_number
    global current_pose_dic

    adjustment_value = data.adjustment_value
    eyebrow_type=data.eyebrow_type
    eye_type = data.eye_type
    mouth_type = data.mouth_type
    adjustment_type=data.adjustment_type
    scale = int(data.scale)
    fps = int(data.fps)
    useOpenCV= data.useOpenCV
    print(eyebrow_type,eye_type,mouth_type,adjustment_type,scale,fps,useOpenCV)
    # adjustment_typeを確認して対応するキーを更新
    if adjustment_type=="eyebrow" or adjustment_type=="eye" or adjustment_type=="iris_small":
        current_pose_dic["eyebrow"]["menue"]=eyebrow_type
        current_pose_dic["eye"]["menue"]=eye_type
        current_pose_dic[adjustment_type]["left"]=adjustment_value
        current_pose_dic[adjustment_type]["right"]=adjustment_value
    elif adjustment_type=="iris_rotation":
        current_pose_dic["iris_rotation"]["x"]=adjustment_value
        current_pose_dic["iris_rotation"]["y"]=adjustment_value
    elif adjustment_type=="mouth":
        current_pose_dic["mouth"]["menue"]=mouth_type
        current_pose_dic["mouth"]["val"]=adjustment_value
    elif adjustment_type=="neck":
        current_pose_dic["neck"]=-adjustment_value
    else: # 'head_x' と 'head_y'
        part, axis = adjustment_type.split("_")
        current_pose_dic[part][axis] = -adjustment_value
    user_id=0
    result_out_image,result = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps)
    if useOpenCV=="On":
        cv2.imshow("Loaded image",result_out_image)
        cv2.waitKey(1)
    else:
        try:
            cv2.destroyWindow("Loaded image")
        except:
            print("Loaded image is not exist")

    h,w,_=result_out_image.shape
    if w>1024:
        out_image = cv2.resize(result_out_image, (1024,int(h*1024/w)))
    else:
        out_image = result_out_image.copy()
    cv2_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
    out_image = Image.fromarray(cv2_image) 
    # PIL Imageをストリームとして処理
    def iterfile():  
        with io.BytesIO() as buffer:
            out_image.save(buffer, format="PNG")
            buffer.seek(0)
            while True:
                chunk = buffer.read(8192)
                if not chunk:
                    break
                yield chunk
    return StreamingResponse(iterfile(), media_type="image/png")



@app.post("/process-emotions/")
async def process_emotions(emotions: str = Form(...), mode: Optional[str] = Form(None), scale: Optional[int] = Form(None), fps: Optional[int] = Form(None),useOpenCV:Optional[str] = Form(None),intensity:Optional[str] = Form(None)):
    global result_out_image
    global img_mode 
    global img_number
    global current_pose_dic

    intensity=float(intensity)
    # ここでemotions_listと他のフォームデータを使用した処理を実装
    print("+++++Value=",emotions,mode,scale,fps,useOpenCV,intensity)
    if emotions=="init":#初期化
        current_pose_dic=Thi.get_init_dic()
        print("====>init=",current_pose_dic)
    elif emotions=="happy":#喜
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="happy_wink"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="iii"
        current_pose_dic["mouth"]["val"]=intensity         
    elif emotions=="angry":#怒
        current_pose_dic["eyebrow"]["menue"]="angry"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="raised_lower_eyelid"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="uuu"
        current_pose_dic["mouth"]["val"]=intensity     
    elif emotions=="sorrow":#哀
        current_pose_dic["eyebrow"]["menue"]="troubled"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="unimpressed"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="ooo"
        current_pose_dic["mouth"]["val"]=intensity 
    elif emotions=="relaxed":#楽
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="relaxed"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="iii"
        current_pose_dic["mouth"]["val"]=1-intensity             
    elif emotions=="smile":#微笑む
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="relaxed"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="aaa"
        current_pose_dic["mouth"]["val"]=intensity              
    elif emotions=="laugh":#笑う
        current_pose_dic["eyebrow"]["menue"]="happy"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="wink"
        current_pose_dic["eye"]["left"]=1-intensity
        current_pose_dic["eye"]["right"]=1-intensity
        current_pose_dic["mouth"]["menue"]="aaa"
        current_pose_dic["mouth"]["val"]=intensity            
    elif emotions=="surprised":#驚く
        current_pose_dic["eyebrow"]["menue"]="lowered"
        current_pose_dic["eyebrow"]["left"]=intensity
        current_pose_dic["eyebrow"]["right"]=intensity
        current_pose_dic["eye"]["menue"]="surprised"
        current_pose_dic["eye"]["left"]=intensity
        current_pose_dic["eye"]["right"]=intensity
        current_pose_dic["mouth"]["menue"]="ooo"
        current_pose_dic["mouth"]["val"]=intensity              
    else:
        print("Emotion Error")
    print(current_pose_dic)
    try:
        cv2_image = np.array(result_image, dtype=np.uint8)
        result_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
    except:
        print("image=cv2")
    if len(mode)>9:  # <= modeがクロップ用の位置情報リストで来た場合。すべての要素が1桁の場合やmodeの文字が10を超えるとだめです
        img_mode = json.loads(mode)
    else:
        img_mode=mode
    user_id=0
    result_out_image,_ = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps=0)
    sleep(0.01)
    result_out_image,_ = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps=0)
    sleep(0.01)
    result_out_image,_ = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps=0)
    sleep(0.01)
    result_out_image,_ = Thi.mp_dic2image_frame(result_out_image,current_pose_dic,img_number,user_id,img_mode,scale,fps=0)

    if useOpenCV=="On":
        cv2.imshow("Loaded image",result_out_image)
        cv2.waitKey(1)
    else:
        try:
            cv2.destroyWindow("Loaded image")
        except:
            print("Loaded image is not exist")
    h,w,_=result_out_image.shape
    if w>1024:
        out_image = cv2.resize(result_out_image, (1024,int(h*1024/w)))
    else:
        out_image = result_out_image.copy()
    cv2_image = cv2.cvtColor(out_image, cv2.COLOR_BGRA2RGBA)
    out_image = Image.fromarray(cv2_image) 
    # 処理された画像をエンコードして返送
    cv2_image = cv2.cvtColor(result_out_image, cv2.COLOR_BGRA2RGBA)
    out_image = Image.fromarray(cv2_image)  
    # PIL Imageをストリームとして処理
    def iterfile():  
        with io.BytesIO() as buffer:
            out_image.save(buffer, format="PNG")
            buffer.seek(0)
            while True:
                chunk = buffer.read(8192)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(iterfile(), media_type="image/png")

#問題のあるコード。動くけどプロセスがうまく噛み合わないのか他の操作ができなくなります。フロントでここに来る
#コードをコメントアウトしています
@app.post("/auto-process/")
def auto_process(test: Optional[str] = Form(None),mode: Optional[str] = Form(None), scale: Optional[int] = Form(None),fps: Optional[int] = Form(None)):
    global result_out_image
    global img_mode 
    global img_number
    global current_pose_dic
    print("--->",test,mode,scale,fps)
    from poser_generater_v1_3_autopose_test import auto_pose_1
    user_id=0
    auto_pose_1(Thi,test,result_out_image,user_id,img_number,img_mode ,scale,fps)

async def rprocess_term():
    #サブプロセスの終了
    Thi.up_scale_proc_terminate()
    Thi.tkh_proc_terminate()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)

