import os
from tqdm import tqdm
import cv2
import random
import numpy as np
import pickle

categories = ["tulipan", "roza", "mniszek", "slonecznik", "stokrotka"]

datadir = './/Praca MGR//flower_photos//Train'
img_size = 50
training_data = []

def create_training_data():
    print("[STATUS] Przygotowanie danych uczących...")
    for category in categories: 
        path = os.path.join(datadir,category)
        class_num = categories.index(category)  
        print("[STATUS] ", class_num+1, "/ 5")
        print("Kategoria (etykieta) : ", category)
        print("Ścieżka do pliku     : ", path)
        print("Numer kategorii      : ", class_num)
        for img in tqdm(os.listdir(path)):
            #print(img)
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
        print(" ")

create_training_data()

create_training_data()
print("[STATUS] Znaleziono ", len(training_data), " obrazów. ")

random.shuffle(training_data)
#print("[STATUS] Przetasowano dane.")

X = []
y = []


for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)

print(X.shape)
test = 1275*50*50*3
print(test)



print(X.shape)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


