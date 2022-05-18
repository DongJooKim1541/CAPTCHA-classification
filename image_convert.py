import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def img_convert():
    file_list=os.listdir("C:/Users/vml/Desktop/Large_Captcha_Dataset")
    for index, name in enumerate(file_list):
        im = Image.open("C:/Users/vml/Desktop/Large_Captcha_Dataset/"+name)
        im_resize = im.resize((int(im.width / 2), int(im.height / 2)))
        im_resize.save("C:/Users/vml/Desktop/Large_captcha_dataset_png_small/"+name.split('.')[0]+".png")
        if index%1000==0:
            print(str(index)+"th png file convert")

# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    img_convert()
    """
    im = Image.open("C:/Users/vml/Desktop/Large_Captcha_Dataset/000AQ.png")
    im_resize = im.resize((int(im.width / 2), int(im.height / 2)))
    im_resize.save("C:/Users/vml/Desktop/000AQ.jpg", quality=20)
    """