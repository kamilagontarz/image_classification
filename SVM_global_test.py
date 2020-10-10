####################
# Import bibliotek #
####################

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import mahotas 
import cv2 
import os
import h5py
from PIL import Image

########################
# Parametry początkowe #
########################

test_img_start = 1

test_files = os.listdir(".//Praca MGR//flower_photos//Test")
num_test_files = len(test_files)
tulip_files = []
dand_files = []
rose_files = []
daisy_files = []
sunf_files = []

for n in range (0, num_test_files):
    if "tulipan" in test_files[n]:
        tulip_files.append(test_files[n])
    if "mniszek" in test_files[n]:
        dand_files.append(test_files[n])
    if "roza" in test_files[n]:
        rose_files.append(test_files[n])
    if "stokrotka" in test_files[n]:
        daisy_files.append(test_files[n])
    if "slonecznik" in test_files[n]:
        sunf_files.append(test_files[n])

tulips = len(tulip_files)
dands = len(dand_files)
roses = len(rose_files)
daisies = len(daisy_files)
sunfs = len(sunf_files)

test_img_stop = {
    "tulipan": tulips+1, 
    "mniszek":dands+1,
    "roza":roses+1,
    "stokrotka":daisies+1,
    "slonecznik":sunfs+1,
}

img_size = tuple((500, 500))
train_path = ".//Praca MGR//flower_photos//Test"
h5_data = './/Praca MGR//Output//data-test.h5'
h5_labels = './/Praca MGR//Output//labels-test.h5'
bins = 8

#######################
# Feature Descriptors #
#######################
# --------------------
# Hu Moments - kształt
# --------------------

def fd_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature 

# ----------------------------
# Haralick Texture - tekstura
# ---------------------------

def fd_haralick(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# ----------------------------
# Color Histogram - kolor
# ---------------------------

def fd_histogram(img, mask=None): 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], [0,1,2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#######################################
# Gromadzenie i zapis cech globalnych #
#######################################

global_features = []
labels = []

class_names = ["tulipan", "mniszek", "roza", "stokrotka", "slonecznik"]

dir = train_path
print("[STATUS] Folder: ", dir)

for c_name in class_names: 
    for x in range(test_img_start,test_img_stop[c_name]):
        img_num = str(x)
        img_name = c_name+img_num+".jpg"
        file = train_path + "/" + img_name
        print("[STATUS] File: ", file)
        file=str(file)
        
        image = cv2.imread(file)
        image = cv2.resize(image, img_size)

        fv_hu_moments = fd_hu_moments(image)     
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
        labels.append(c_name)
        global_features.append(global_feature)

print("[STATUS] Zakończono poszukiwanie cech globalnych.")
print("[STATUS] Rozmiar wektora: {}".format(np.array(global_features).shape))

encoded_labels = LabelEncoder()
target = encoded_labels.fit_transform(labels)

scale = MinMaxScaler(feature_range=(0, 1))
scaled_features = scale.fit_transform(global_features)

open(h5_data,"w+") 
open(h5_labels,"w+")

print("[STATUS] Stworzono pliki data-test.h5 i label-test.h5.")

h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(scaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()
print("[STATUS] Zakończono zapis danych.")
