####################
# Import bibliotek #
####################

import h5py
import numpy as np
import os
import glob
#from sklearn.externals import joblib
from sklearn import svm
from sklearn import metrics

print("\n\n")
print("[STATUS] Trenowanie...")

#################################
# Wczytanie parametrów z plików #
#################################

h5_data_train    = './/Praca MGR//Output//data.h5'
h5_labels_train  = './/Praca MGR//Output//labels.h5'
h5_data_test    = './/Praca MGR//Output//data-test.h5'
h5_labels_test  = './/Praca MGR//Output//labels-test.h5'

h5f_data_train  = h5py.File(h5_data_train, 'r')
h5f_label_train = h5py.File(h5_labels_train, 'r')
h5f_data_test  = h5py.File(h5_data_test, 'r')
h5f_label_test = h5py.File(h5_labels_test, 'r')

global_features_string_train = h5f_data_train['dataset_1']
global_labels_string_train = h5f_label_train['dataset_1']
global_features_string_test = h5f_data_test['dataset_1']
global_labels_string_test = h5f_label_test['dataset_1']

global_features_train = np.array(global_features_string_train)
global_labels_train = np.array(global_labels_string_train)
global_features_test = np.array(global_features_string_test)
global_labels_test = np.array(global_labels_string_test)

h5f_data_train.close()
h5f_label_train.close()
h5f_data_test.close()
h5f_label_test.close()

trainDataGlobal=global_features_train
trainLabelsGlobal=global_labels_train
testLabelsGlobal=global_labels_test
testDataGlobal=global_features_test

print("[STATUS] Dane przygotowane")
print("         Dane uczące              : {}".format(trainDataGlobal.shape))
print("         Dane testujące           : {}".format(testDataGlobal.shape))
print("         Klasy danych uczących    : {}".format(trainLabelsGlobal.shape))
print("         Klasy danych testujących : {}".format(testLabelsGlobal.shape))

################################
# Przygotowanie klasyfikatorów #
################################

clf  = svm.SVC(kernel='linear') 
print("[STATUS] Stworzono klasyfikator SVM")

clf.fit(trainDataGlobal, trainLabelsGlobal)
print("[STATUS] Trenowanie modelu zakończone")

print("[STATUS] Predykcja...")
predLabelsGlobal = clf.predict(testDataGlobal)

data_num = len(testLabelsGlobal)
correctly_tested = 0
incorrectly_tested = 0

for i in range(0,data_num):
    current_test = testLabelsGlobal[i]
    current_pred = predLabelsGlobal[i]
    
    if(current_test == current_pred):
        correctly_tested += 1
    else: 
        incorrectly_tested += 1


print("[STATUS] Predykcja zakończona")
print("         Dokładność (accuracy):",metrics.accuracy_score(testLabelsGlobal, predLabelsGlobal))
print("         Poprawnie sklasyfikowane: ", correctly_tested)
print("         Niepoprawnie sklasyfikowane: ", incorrectly_tested)
