import os
import glob
import datetime
import tarfile
import urllib.request

all_classes = 0

if not os.path.exists("Praca MGR//flower_photos" + "//" + "tulips"):
        print("Folder ","Praca MGR//flower_photos" + "//" + "tulips"," nie istnieje.")
else: 
    all_classes += 1
    all_tulips = len(os.listdir("Praca MGR//flower_photos" + "//" + "tulips"))
    test_tulips = int(0.1*all_tulips)
    train_tulips = all_tulips - test_tulips

    print("******************************************************")
    print("*                     TULIPANY                       *")
    print("* Wszystkie: ", all_tulips, " | Testowe: ", test_tulips, " | Trenujące: ", train_tulips, "*")
    print("******************************************************")

if not os.path.exists("Praca MGR//flower_photos" + "//" + "roses"):
        print("Folder ","Praca MGR//flower_photos" + "//" + "roses"," nie istnieje.")
else:
    all_classes += 1
    all_roses = len(os.listdir("Praca MGR//flower_photos" + "//" + "roses"))
    test_roses = int(0.1*all_roses)
    train_roses = all_roses - test_roses

    print("******************************************************")
    print("*                       RÓŻE                         *")
    print("* Wszystkie: ", all_roses, " | Testowe: ", test_roses, " | Trenujące: ", train_roses, "*")
    print("******************************************************")

if not os.path.exists("Praca MGR//flower_photos" + "//" + "daisy"):
        print("Folder ","Praca MGR//flower_photos" + "//" + "daisy"," nie istnieje.")
else: 
    all_classes += 1
    all_daisy = len(os.listdir("Praca MGR//flower_photos" + "//" + "daisy"))
    test_daisy = int(0.1*all_daisy)
    train_daisy = all_daisy - test_daisy
    
    print("******************************************************")
    print("*                     STOKROTKI                      *")
    print("* Wszystkie: ", all_daisy, " | Testowe: ", test_daisy, " | Trenujące: ", train_daisy, "*")
    print("******************************************************")

if not os.path.exists("Praca MGR//flower_photos" + "//" + "dandelion"):
        print("Folder ","Praca MGR//flower_photos" + "//" + "dandelion"," nie istnieje.")

else: 
    all_classes += 1
    all_dand = len(os.listdir("Praca MGR//flower_photos" + "//" + "dandelion"))
    test_dand = int(0.1*all_dand)
    train_dand = all_dand - test_dand

    print("******************************************************")
    print("*                      MNISZKI                       *")
    print("* Wszystkie: ", all_dand, " | Testowe: ", test_dand, " | Trenujące: ", train_dand, "*")
    print("******************************************************")

if not os.path.exists("Praca MGR//flower_photos" + "//" + "sunflowers"):
        print("Folder ","Praca MGR//flower_photos" + "//" + "sunflowers"," nie istnieje.")

else: 
    all_classes += 1
    all_sunf = len(os.listdir("Praca MGR//flower_photos" + "//" + "sunflowers"))
    test_sunf = int(0.1*all_sunf)
    train_sunf = all_sunf - test_sunf

    print("******************************************************")
    print("*                    SŁONECZNIKI                     *")
    print("* Wszystkie: ", all_sunf, " | Testowe: ", test_sunf, " | Trenujące: ", train_sunf, "*")
    print("******************************************************")

    all_dirs =  ["tulips",  "roses", "daisy",     "dandelion", "sunflowers"]
    all_names = ["tulipan", "roza",  "stokrotka", "mniszek",   "slonecznik"]
    all_list = [all_tulips, all_roses, all_daisy, all_dand, all_sunf]
    all_test = [test_tulips, test_roses, test_daisy, test_dand, test_sunf]
    all_train = [train_tulips, train_roses, train_daisy, train_dand, train_sunf]

if(all_classes!=5):
   print("[ERROR] Foldery nie istnieją.")
else: 
    for i in range (0,all_classes):
        current_dir = all_dirs[i]
        current_class = all_names [i]
        current_all_pics = all_list[i]
        current_all_test = all_test[i]
        current_all_train = all_train[i]

        for tr in range (1, current_all_train+1):
            src_dir_path = "Praca MGR//flower_photos" + "//" + current_dir
            files = os.listdir(path=src_dir_path)
            current_file = files[0]
            src = src_dir_path + "//" + current_file
        
            if not os.path.exists(".//Praca MGR//flower_photos//Train//"+current_class):
                os.makedirs(".//Praca MGR//flower_photos//Train//"+current_class)
                print("[INFO] Stworzono folder 'Train "+ current_class)

            dst = "Praca MGR//flower_photos//Train//"+current_class+"//"+current_class+str(tr)+".jpg"

            train_files_num = len(os.listdir("Praca MGR//flower_photos//Train//"+current_class))
            if train_files_num != current_all_train+1: 
                os.rename(src, dst)       

        for tst in range (1, current_all_test+1):
            src_dir_path = "Praca MGR//flower_photos" + "//" + current_dir
            files = os.listdir(path=src_dir_path)
            current_file = files[0]
            src = src_dir_path + "//" + current_file
        
            if not os.path.exists(".//Praca MGR//flower_photos//Test//"):
                os.makedirs(".//Praca MGR//flower_photos//Test//")
                print("[INFO] Stworzono folder 'Test")

            dst = "Praca MGR//flower_photos//Test//"+"//"+current_class+str(tst)+".jpg"
            os.rename(src, dst)

    for di in all_dirs:
        old_dir_path = "Praca MGR//flower_photos" + "//" + di
        if not os.path.exists(old_dir_path):
            print("Folder ",old_dir_path," nie istnieje.")
        else: 
            os.rmdir(old_dir_path)
            print("Usunięto ", old_dir_path, ".")

