from PIL import Image
import os 
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from skimage.feature import local_binary_pattern
import cv2
from itertools import zip_longest
import numpy as np



def image_to_feature(df:pd.DataFrame,img,full_path, features_func):
    rows = []
    
    data = {"animal":[]}
    
    if "Cat" in full_path:

        animal =  1 #"Cat"

    else: animal = 0 #"Dog"

    
    imgreadre = cv2.resize(img,(300,280))
    gray = cv2.cvtColor(imgreadre, cv2.COLOR_BGR2GRAY)
    
    data["file"] = full_path
    data["animal"] = animal
    
    features = features_func(gray)

    for i, f in enumerate(features):

        data[f"f_{i}"] = f

    dfcopy = pd.DataFrame([data])
    df = pd.concat([df,dfcopy], ignore_index=True)

    return df

def is_valid_image(path):
    try:

        with Image.open(path) as img:
            img.verify()  # checks file integrity
        
        return True
    
    
    except:

        return False
    
def img_to_csv_data(animal, path, loops_unitl_save=5, list_of_feature=None):
    folder = path + animal
    desc = {}
    
    for name, feature_func in list_of_feature.items():
        print(f"{animal} - {name}")
        df = pd.DataFrame()
        corrupt_amount = 0
        nr = 0

        for file in tqdm(sorted(os.listdir(folder))):
            
            full_path = os.path.join(folder, file)

            if not file.endswith(".jpg"):
                continue

            if not is_valid_image(full_path):
                corrupt_amount += 1
                continue
            

            img = cv2.imread(full_path)
            if img is None:
                continue

            if img.shape[0] < 300 or img.shape[1] < 280:
                corrupt_amount += 1
                continue
            
            df = image_to_feature(df,img,full_path, feature_func)
            
            nr += 1
            if nr >= loops_unitl_save:
                df.to_csv(
                    f"data/{animal}_{name}.csv",
                    mode="a",
                    header=not os.path.exists(f"data/{animal}_{name}.csv"),
                    index=False
                )
                df = pd.DataFrame()
                nr = 0
        print(f"done with {animal} - {name}")
        # after for file loop ends
        if not df.empty:
            df.to_csv(
                f"data/{animal}_{name}.csv",
                mode="a",
                header=not os.path.exists(f"data/{animal}_{name}.csv"),
                index=False
            )
        key = f"{name}{animal}"   # e.g. HOGCat
    
        desc[key] = {
            "animal": animal,
            "feature": name,
            "feature_func": feature_func,
            "corrupt": corrupt_amount
        }
        df_data_preproc_info = pd.DataFrame(desc)
        df_data_preproc_info.to_csv("data/data_preproccesing_information.csv")
        print(desc)


def run_pipeline(animals = ["Dog","Cat"],
    path = "data/PetImages/",list_of_feature = None):

    for ani in animals:
        print(ani)
        img_to_csv_data(animal= ani,path=path,list_of_feature = list_of_feature,loops_unitl_save=500)




def interleave_csv(cat_path, dog_path, output_path, chunksize=50):
    cat_reader = pd.read_csv(cat_path, chunksize=chunksize)
    dog_reader = pd.read_csv(dog_path, chunksize=chunksize)

    first = True

    for cat_chunk, dog_chunk in tqdm(zip_longest(cat_reader, dog_reader)):
        frames = []
        print("cycle start")
        
        if cat_chunk is not None:
            
            cat_chunk = cat_chunk.rename(columns={"0": "animal"})
            cat_chunk = cat_chunk.sample(frac=1)
            frames.append(cat_chunk)

        if dog_chunk is not None:
              
            dog_chunk = dog_chunk.rename(columns={"0": "animal"})
            dog_chunk = dog_chunk.sample(frac=1)
            frames.append(dog_chunk)

        if not frames:
            continue

        combined = pd.concat(frames)
        combined = combined.sample(frac=1)

        combined.to_csv(
            output_path,
            mode="a",
            header=first,
            index=False
        )

        first = False

def extract_HOG(gray):
    return hog(
    gray,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    feature_vector=True)

def extract_lbp(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=59,
        range=(0, 59))
    
    hist = hist / (hist.sum() + 1e-6)
    return hist





def merge_csv(path1, path2, output_path, name1, name2):

    if os.path.exists(output_path):
        os.remove(output_path)

    reader1 = pd.read_csv(path1, chunksize=20000, low_memory=False)
    reader2 = pd.read_csv(path2, chunksize=20000, low_memory=False)

    first = True

    for chunk1, chunk2 in tqdm(zip(reader1, reader2)):

        chunk1 = chunk1.rename(columns=lambda x: f"{name1}_{x}" if x.startswith("f_") else x)
        chunk2 = chunk2.rename(columns=lambda x: f"{name2}_{x}" if x.startswith("f_") else x)

        chunk2 = chunk2.drop(columns=["file", "animal"], errors="ignore")

        combined = pd.concat([chunk1, chunk2], axis=1)

        combined.to_csv(
            output_path,
            mode="a",
            header=first,
            index=False
        )

        first = False
if __name__ == "__main__":
    print("starting ------------------------------------------------------------")
    list_of_feature_methods = {"HOG" : extract_HOG,"LBP":extract_lbp }
    exct_list = [["data/Cat_HOG.csv","data/Cat_LBP.csv","data/Cat_HOG_LBP.csv","HOG","LBP"]
                 ,["data/Dog_HOG.csv","data/Dog_LBP.csv","data/Dog_HOG_LBP.csv","HOG","LBP"]]
    for item in exct_list:
        merge_csv(item[0],item[1],item[2],item[3],item[4])
    print("starting pipeline ----------------------------------------------------------------")
    interleave_csv("data/Cat_HOG_LBP.csv","data/Dog_HOG_LBP.csv","data/shuffledHOG_LBP.csv",chunksize=50000)
    
    