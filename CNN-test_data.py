import os
from tqdm import tqdm
import cv2
import random
import numpy as np
import pickle

categories = ["tulipan", "roza", "mniszek", "slonecznik", "stokrotka"]
classes = {"tulipan":0, "roza":1, "mniszek":2, "slonecznik":3, "stokrotka":4}
datadir = './/Praca MGR//flower_photos//Test'
img_size = 50
test_data = []

def create_testing_data():
    print("[STATUS] Przygotowanie danych testujących...")
    path = datadir
        
    for img in tqdm(os.listdir(path)):
        img_p = os.path.join(path,img)   
        if "tulipan" in img_p:
          class_num = classes["tulipan"]
        if "roza" in img_p:
            class_num = classes["roza"]
        if "mniszek" in img_p:
            class_num = classes["mniszek"]
        if "slonecznik" in img_p:
            class_num = classes["slonecznik"]
        if "stokrotka" in img_p:
            class_num = classes["stokrotka"]
        print(img_p)
        try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
                test_data.append([new_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
                pass
        print(" ")

create_testing_data()

create_testing_data()
print("[STATUS] Znaleziono ", len(test_data), " obrazów. ")

random.shuffle(test_data)
#print("[STATUS] Przetasowano dane.")

X = []
y = []

for features,label in test_data:
    X.append(features)
    y.append(label)

X = np.array(X)

pickle_out = open("X-test.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y-test.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

