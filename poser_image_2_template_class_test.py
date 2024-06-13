import time
import os
import sys
import numpy as np
from PIL import Image
import torch
from datetime import datetime
import requests

import shutil
import glob
import socket
import pickle
import argparse
import cv2

from tkh_up_scale import upscale

from poser_image_2_template_class import Image2form

def main():
    parser = argparse.ArgumentParser(description='Talking head')
    parser.add_argument("--up_url",   type=str,  default="http://0.0.0.0:8008/resr_upscal/", help="サービスを提供するポートを指定。")
    parser.add_argument('--filename','-i', default='tumblr_o51k7aQC5Z1qa63ddo1_1280.jpg', type=str)
    parser.add_argument('--shape','-s', default='face', type=str)

    args = parser.parse_args()
    filename =args.filename
    img_shape =args.shape
    up_url= args.up_url
    print("up_url=",up_url)

    I2f=Image2form(up_url)
                
    #任意の入力画像からTalking-Head用画像を生成、remrgb使用、白黒2値のマスクを用いた改良が必要、haikeiバージョンも必要
    print("test7")
    pil_input_image = Image.open(filename)
        
    result , pil_w_img = I2f.image_data_form(pil_input_image,"pil")

    pil_w_img.show()

if __name__ == "__main__":
    main()
