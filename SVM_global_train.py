####################
# Import bibliotek #
####################

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import mahotas ## conda install mahotas
import cv2 ## pip install opencv-python 
import os
import h5py
from PIL import Image

########################
# Parametry początkowe #
########################

if not os.path.exists(".//Praca MGR//Output"):
    os.makedirs(".//Praca MGR//Output")
    print("[INFO] Stworzono folder 'Output'")
else:
    print("[INFO] Folder 'Output' już istnieje")

train_tulips = len(os.listdir(".//Praca MGR//flower_photos//Train//tulipan"))
train_dand = len(os.listdir(".//Praca MGR//flower_photos//Train//mniszek"))
train_rose = len(os.listdir(".//Praca MGR//flower_photos//Train//roza"))
train_daisy = len(os.listdir(".//Praca MGR//flower_photos//Train//stokrotka"))
train_sunf = len(os.listdir(".//Praca MGR//flower_photos//Train//slonecznik"))

img_per_folder = {
    "tulipan" : train_tulips,
    "mniszek" : train_dand,
    "roza" : train_rose,
    "slonecznik" : train_sunf,
    "stokrotka" : train_daisy
}

img_size = tuple((500, 500))
train_path = ".//Praca MGR//flower_photos//Train//"
h5_data = './/Praca MGR//Output//data.h5'
h5_labels = './/Praca MGR//Output//labels.h5'
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

train_dirs = os.listdir(train_path)
train_dirs.sort()
print("[INFO] Znaleziono ", len(train_dirs), " folderów.")

global_features = []
labels = []
dir_count = 0

for training_dir in train_dirs:
    dir_count +=1
    dir = train_path + "/" + training_dir
    current_label = training_dir
    print("[STATUS] ", dir_count, "/ 5 - Aktualny folder: ", dir)
    for x in range(1,img_per_folder[training_dir]+1):
        file = dir + "/" + current_label + str(x) + ".jpg"
        image = cv2.imread(file)
        image = cv2.resize(image, img_size)
        fv_hu_moments = fd_hu_moments(image)     
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
        labels.append(current_label)
        global_features.append(global_feature)

print("[STATUS] Zakończono poszukiwanie cech globalnych.")
print("[STATUS] Rozmiar wektora: {}".format(np.array(global_features).shape))

encoded_labels = LabelEncoder()
target = encoded_labels.fit_transform(labels)

scale = MinMaxScaler(feature_range=(0, 1))
scaled_features = scale.fit_transform(global_features)

open(h5_data,"w+") 
open(h5_labels,"w+")

print("[STATUS] Stworzono pliki data.h5 i label.h5.")

h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(scaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()
print("[STATUS] Zakończono zapis danych.")
    
