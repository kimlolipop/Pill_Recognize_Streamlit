import torch
# import matplotlib.pyplot as plt
import requests
import io

from keras.applications.imagenet_utils import preprocess_input
import cv2
import numpy as np
import pandas as pd
import seaborn
import os

from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import normalize
import pickle

# from PIL import Image
# from keras.preprocessing import image


# try:
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# except:
    # pass

# Yolo

def load_model():
  model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='weight/cannyS.pt')
  # model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='/content/drive/MyDrive/Super AI Engineer/ฝึกงาน/Pill Detection/data_yolo/modelS.pt')

  return model

def resize_im(img):

  # im = cv2.imread(pathh)
  # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  ht, wd, cc= img.shape

  # create new image of desired size and color (blue) for padding
  ww = 1080
  hh = 1080
  color = (0,0,0)
  result = np.full((hh,ww,cc), color, dtype=np.uint8)

  # compute center offset
  xx = (ww - wd) // 2
  yy = (hh - ht) // 2

  # copy img image into center of result image
  result[yy:yy+ht, xx:xx+wd] = img

  return result


def square(img):

  shape = img.shape[0], img.shape[1]
  # print(shape)  
  min_shape = min(shape)

  if min_shape == shape[0]:
    ww = round((shape[1] - shape[0]) / 2)
    result = img[0:min_shape, ww:ww+min_shape]

  else:
    hh = round((shape[0] - shape[1]) / 2)
    result = img[hh:hh+ min_shape, 0:min_shape]
    # print(hh)



  return result


def read_im(path):
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img



def pre_process(img): # preprocess_img

  img = square(img)


  if img.shape[0] <= 1080:
    img = resize_im(img)
  else: 
    img = cv2.resize(img, (1080 , 1080))

  im = img

  #canny 
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = cv2.resize(img, (1080 , 1080))
  img = cv2.GaussianBlur(img,(5,5),0)
  img = cv2.bilateralFilter(img,9,75,75)
  img = cv2.GaussianBlur(img,(5,5),0)
  img = cv2.Canny(img,10,10)
  img = cv2.dilate(img,(20,20),iterations = 3)

  return img, im

def localize(img):
  results = yolo(img)
  # print(results.xyxy[0])
  return results

def crop_localize(results, im):

  local_im = []

  d = results.xyxy[0]
  col = ['x1','y1','x2','y2','confidence','class']
  df2 = pd.DataFrame(np.array(d), columns=col)
  for i in range(0, len(df2)):

    x1 = int(df2[['x1']].iloc[i].values[0])
    y1 = int(df2[['y1']].iloc[i].values[0])
    x2 = int(df2[['x2']].iloc[i].values[0])
    y2 = int(df2[['y2']].iloc[i].values[0])

    local_im.append(im[y1:y2, x1:x2])

  return local_im


#end Yolo

# Classification
def load_model_ex():
  extract = ResNet50(include_top=False, weights='imagenet', classes=1000)

  with open('weight\svm_10_class.pkl', 'rb') as f:
    clf = pickle.load(f)


  return extract, clf



def square_clss(img, dim=(224,224)):
  # img = cv2.imread(path)
  shape = img.shape[0], img.shape[1]
  # print(shape)
  min_shape = min(shape)

  if min_shape == shape[0]:
    ww = round((shape[1] - shape[0]) / 2)
    result = img[0:min_shape, ww:ww+min_shape]

  else:
    hh = round((shape[0] - shape[1]) / 2)
    result = img[hh:hh+ min_shape, 0:min_shape]
    # print(hh)

  
  result = cv2.resize(result, dim)

  return result


def extract_feature(img):

  
  img = img/1.0
  img = img.astype(np.float32)
  # img = img[...,::-1].astype(np.float32)
 
  img = square_clss(img, (224,224))

  x = image.img_to_array(img) 
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  features = extract.predict(x, batch_size=1,verbose=0)
  features = np.ndarray.flatten(features).astype('float64')
  feat = normalize([features])[0]


  return feat


def create_fea(local_im):
  clsset = pd.DataFrame()
  fea = []

  for i in range(0, len(local_im)):
    a = extract_feature(local_im[i])
    fea.append(a)

  clsset['feature'] = fea


  return clsset


def predict_ndc(dataset):

  fea = dataset['feature']
  ans = clf.predict(np.vstack(fea.values))

  return ans







#main
# path = 'img/test.jpg'


# main
# img = read_im(path)


def main_classify(img):

  img, im = pre_process(img)
  result = localize(img)
  local_im = crop_localize(result, im)



  # extract, clf = load_model_ex()
  feavec = create_fea(local_im)


  ans = predict_ndc(feavec)

  print(local_im)

  return ans, local_im



def main():
  #main
  path = 'img/test.jpg'


  # main
  img = read_im(path)
  ans = main_classify(img)
  print(ans)
  # return ans

extract, clf = load_model_ex()
yolo = load_model()
# main()
