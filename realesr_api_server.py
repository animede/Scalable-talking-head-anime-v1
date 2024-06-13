import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from datetime import datetime
import pickle
from fastapi import FastAPI, File, UploadFile, Form
from starlette.responses import Response,StreamingResponse
from io import BytesIO
import numpy as np
import time

# ++++++++++++++  up scale ++++++++++++++++
def  up_scale(img , scale):
        print("inf_start_time=",datetime.now())
        global upsampler
        try:
            output, _ = upsampler.enhance(img , outscale=scale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        print("inf_end_time=",datetime.now())
        return output

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
parser.add_argument('-n','--model_name', type=str, default='realesr-animevideov3', help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | realesr-animevideov3 | realesr-general-x4v3'))
parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
parser.add_argument('-dn','--denoise_strength',type=float, default=0.5, help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. Only used for the realesr-general-x4v3 model'))
parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
parser.add_argument( '-t', '--test', type=bool, default=False, help='excecute test PG if True')
parser.add_argument("--host", type=str,  default="0.0.0.0",  help="サービスを提供するip アドレスを指定。")
parser.add_argument("--port", type=int,  default=50008,    help="サービスを提供するポートを指定。")
args = parser.parse_args()

# determine models according to model names
args.model_name = args.model_name.split('.')[0]
if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        #model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        #model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4

 #+++++++++++++++++++  init  +++++++++++++++++++
model_path = "./weights/" + args.model_name +".pth"
print(model_path )
#print(netscale)
# use dni to control the denoise strength
dni_weight = None
if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
    # restorer
upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0)

#+++++++++++++++++++ TEST +++++++++++++++++++
if args.test==True:
        if os.path.isfile(args.input):
            paths = [args.input]
        else:
            paths = sorted(glob.glob(os.path.join(args.input, '*')))
        img_list=[]
        for idx, path in enumerate(paths):
            imgname, extension = os.path.splitext(os.path.basename(path))
            print('Testing', idx, imgname)
            cv_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img_list.append(cv_img)
        print("start_time=",datetime.now())
        count=len(img_list)
        for i in range(0,count):
            img=img_list[i]
            output = up_scale(img , args.outscale)
            #output = cv2.resize(output,(1024,1024 ))
            if len(img.shape) == 3 and img.shape[2] == 4:
                extension = '.png'
            else:
                extension = '.jpg'
            save_path = "./results/" + args.output+ str(i)+extension
            cv2.imwrite(save_path, output) #if files are require
        print("end_time=",datetime.now())

# =============    FastAPI  ============
app = FastAPI()

@app.post("/resr_upscal/")
def resr_upscal(image: UploadFile = File(...),scale:int= Form(...)): #file=OpenCV
    #print("Recive_time0=",datetime.now())
    print("scale=",scale)
    start_now=time.time()
    image_data = image.file.read()
    img =(pickle.loads(image_data))
    out_img, _ = upsampler.enhance(img, outscale=scale)
    frame_data = pickle.dumps(out_img, 5)  # tx_dataはpklデータ、イメージのみ返送
    print("Upscale time=",(time.time()-start_now)*1000,"mS")
    #print("send_time=",datetime.now())
    return Response(content=frame_data, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

