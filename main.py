import torch
import requests
import io

from keras.applications.imagenet_utils import preprocess_input
import cv2
import numpy as np
import pandas as pd
import seaborn
import os

import tensorflow as tf
from tensorflow import keras
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet101 import ResNet101
# from keras.applications.resnet152 import ResNet152
# from keras.applications.vgg19 import VGG19
# from keras.applications.efficientnetb3 import EfficientNetB3

# from keras_resnet.models import ResNet50 #, ResNet101, ResNet152, VGG19, EfficientNetB3

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import normalize
import pickle



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Yolo
def load_model(): # load yolo
  model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='weight/cannyS.pt')

  return model


def read_im(path): # read img to cv2 (1)
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img


def pre_process(img): # preprocess_img (2)

  img = square(img)


  if img.shape[0] <= 1080:
    # img = resize_im(img)
    cv2.resize(img, (1080 , 1080))
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


# Module pre-process ############################################### (2)
def square(img): # crop img rectangle shape to square shape

  shape = img.shape[0], img.shape[1]
  min_shape = min(shape)

  if min_shape == shape[0]:
    ww = round((shape[1] - shape[0]) / 2)
    result = img[0:min_shape, ww:ww+min_shape]

  else:
    hh = round((shape[0] - shape[1]) / 2)
    result = img[hh:hh+ min_shape, 0:min_shape]

  return result

def resize_im(img): # Conditional module --> # use extra this module when img < 1080 Px (padding img ans fill RGB(0,0,0)) 

  ht, wd, cc= img.shape

  # create new image of desired size and color (black) for padding

  #Can edit........................
  ww = 1080 # result padding size
  hh = 1080 # result padding size
  color = (0,0,0) # filll color
  # ................................

  result = np.full((hh,ww,cc), color, dtype=np.uint8)

  # compute center offset
  xx = (ww - wd) // 2
  yy = (hh - ht) // 2

  # copy img image into center of result image
  result[yy:yy+ht, xx:xx+wd] = img

  return result
####################################################################


def localize(img): # run Yolo
  results = yolo(img)
  
  return results


def crop_localize(results, im): #use output from yolo to crop image
  local_im = []
  d = results.xyxy[0]
  col = ['x1','y1','x2','y2','confidence','class']
  df2 = pd.DataFrame(np.array(d), columns=col)

  if len(df2) > 0:
    
    for i in range(0, len(df2)):

      x1 = int(df2[['x1']].iloc[i].values[0])
      y1 = int(df2[['y1']].iloc[i].values[0])
      x2 = int(df2[['x2']].iloc[i].values[0])
      y2 = int(df2[['y2']].iloc[i].values[0])

      local_im.append(im[y1:y2, x1:x2])

    # print(local_im)
    return local_im

  else:
    local_im.append(im)
    # print(im)
    return local_im
#end Yolo



# Classification
def load_model_ex(): # load model extract feature
  extract_resnet50 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', classes=1000)
  extract_resnet101 = tf.keras.applications.ResNet101(include_top=False, weights='imagenet', classes=1000)
  extract_VGG19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', classes=1000)

  with open('weight/resnet101_200_class.pkl', 'rb') as f:
    clf_resnet50 = pickle.load(f)


  with open('weight/resnet101_200_class.pkl', 'rb') as f:
    clf_resnet101 = pickle.load(f)


  with open('weight/VGG19_200_class.pkl', 'rb') as f:
    clf_vgg19 = pickle.load(f)

  return extract_resnet50, extract_resnet101, extract_VGG19,clf_resnet50, clf_resnet101, clf_vgg19



def square_clss(img, dim=(224,224)): # resize to 224 and crop to square
  
  shape = img.shape[0], img.shape[1]
  min_shape = min(shape)

  if min_shape == shape[0]:
    ww = round((shape[1] - shape[0]) / 2)
    result = img[0:min_shape, ww:ww+min_shape]

  else:
    hh = round((shape[0] - shape[1]) / 2)
    result = img[hh:hh+ min_shape, 0:min_shape]

  
  result = cv2.resize(result, dim)

  return result


def extract_feature(img): # run extract feature

  #pre process
  img = img/1.0
  img = img.astype(np.float32)
 
  img = square_clss(img, (224,224))

  x = image.img_to_array(img) 
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  # extract feature resnet50
  features = extract_resnet50.predict(x, batch_size=1,verbose=0) # DL extract feature
  features = np.ndarray.flatten(features).astype('float64') # dense
  feat_resnet50 = normalize([features])[0] # norm

  # extract feature resnet101
  features = extract_resnet101.predict(x, batch_size=1,verbose=0) # DL extract feature
  features = np.ndarray.flatten(features).astype('float64') # dense
  feat_resnet101 = normalize([features])[0] # norm


  # extract feature VGG19
  features = extract_VGG19.predict(x, batch_size=1,verbose=0) # DL extract feature
  features = np.ndarray.flatten(features).astype('float64') # dense
  feat_VGG19 = normalize([features])[0] # norm


  return feat_resnet50, feat_resnet101, feat_VGG19 #add 5 feat


def create_fea(local_im): # take each img to extract feature
  clsset = pd.DataFrame()
  fea_resnet50 = []
  fea_resnet101 = []
  # fea_resnet152 = []
  fea_VGG19 = []
  # fea_effnetB3 = []

  for i in range(0, len(local_im)):
    resnet50, resnet101, VGG19 = extract_feature(local_im[i])



    fea_resnet50.append(resnet50)
    fea_resnet101.append(resnet101)
    fea_VGG19.append(VGG19)


  

  clsset['resnet50'] = fea_resnet50
  clsset['resnet101'] = fea_resnet101
  clsset['VGG19'] = fea_VGG19



  return clsset


def predict_ndc(dataset): # classify model

  ##### edit + Stacking
  fea = dataset['resnet50']
  ans1 = clf_resnet50.predict(np.vstack(fea.values))

  
  fea = dataset['resnet101']
  ans2 = clf_resnet101.predict(np.vstack(fea.values))

  

  fea = dataset['VGG19']
  ans4 = clf_vgg19.predict(np.vstack(fea.values))
  

  return ans1, ans2, ans4



def main_classify(img): # run all process

  # localize
  img, im = pre_process(img)
  result = localize(img)
  local_im = crop_localize(result, im)

  # classify
  feavec = create_fea(local_im)
  ans1, ans2, ans4  = predict_ndc(feavec)





  # voting
  ans = []
  for z in range(0, len(ans1)):
    test_list = [ans1[z], ans2[z], ans4[z]]
    # test_list = ['9', '4', '5', '4', '4', '5', '9', '5', '4']
    
    # printing original list
    print ("Original list : " + str(test_list))
      
    # using naive method to 
    # get most frequent element
    max = 0
    res = test_list[0]
    for i in test_list:
        freq = test_list.count(i)
        if freq > max:
            max = freq
            res = i

    ans.append(res)
  # print ("Most frequent number is : " + str(res))



  return ans, local_im



#Load Weight Model
extract_resnet50, extract_resnet101, extract_VGG19, clf_resnet50, clf_resnet101, clf_vgg19 = load_model_ex()
yolo = load_model()