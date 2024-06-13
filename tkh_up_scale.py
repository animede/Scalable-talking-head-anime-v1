import time
from time import sleep
import numpy as np
import cv2
from PIL import Image
import argparse
import pickle
import requests


#PIL形式の画像を動画として表示
def image_show(imge):
    imge = np.array(imge)
    imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Loaded image",imge)
    cv2.waitKey(1)

# ++++++++++++++  up scale ++++++++++++++++
def up_scale(url , img ,  scale=4):
    #_, img_encoded = cv2.imencode('.jpg', img)
    images_data = pickle.dumps(img, 5) 
    files = {"image": ("img.dat",  images_data, "application/octet-stream")}
    data = {"scale": scale}
    response = requests.post(url, files=files,data=data)
    
    all_data =response.content
    up_data = (pickle.loads(all_data))#元の形式にpickle.loadsで復元
    return up_data #形式はimg_mode指定の通り

def main():

    print("TEST")
    
    parser = argparse.ArgumentParser(description='Talking Head')
    parser.add_argument('--filename','-i', default='000002.png', type=str)
    parser.add_argument('--mode', default="full", type=str)#full,breastup,waistup,upperbody
    parser.add_argument('--scale', default=4, type=int)#2,4,8
    parser.add_argument("--host", type=str,  default="0.0.0.0",  help="サービスを提供するip アドレスを指定。")
    parser.add_argument("--port", type=int,  default=8008,    help="サービスを提供するポートを指定。")
    args = parser.parse_args()

    host="0.0.0.0"    # サーバーIPアドレス定義
    port=8008          # サーバー待ち受けポート番号定義
    url="http://" + host + ":" + str(port) + "/resr_upscal/"
    
    mode = args.mode
    scale= args.scale
    print("upscale=",mode,"scale=",scale)
    filename =args.filename
    print("filename=",filename)

    image = Image.open(filename)#image=512x512xαチャンネル
    imge = np.array(image)
    cv2_imge = cv2.cvtColor(imge, cv2.COLOR_RGBA2BGRA)

    upscale_image = upscale(url ,cv2_imge, mode, scale)

    cv2.imshow("Loaded image",upscale_image)
    cv2.waitKey(1000)

def crop_image(image, top, left, height, width):
    # 画像を指定された位置とサイズで切り出す
    cropped_image = image[top:top+height, left:left+width]
    return cropped_image

def upscale(url ,image, mode, scale):
    print("xxxxxxxxxxxxxxxxxxxxxxx  mode=",mode)
    if mode=="breastup":
        cropped_image = crop_image(image, top=55, left=128, height=256, width=256)
    elif mode=="waistup":
        cropped_image = crop_image(image, top=55, left=128, height=290, width=256)
    elif mode=="upperbody":
        cropped_image = crop_image(image, top=55, left=143, height=336, width=229)
    elif mode=="full":
        cropped_image = image
    else:
        cropped_image = crop_image(image, top=mode[0], left=mode[1], height=mode[2], width=mode[3])        
    return up_scale(url , cropped_image ,  scale)
            
if __name__ == "__main__":
    main()
