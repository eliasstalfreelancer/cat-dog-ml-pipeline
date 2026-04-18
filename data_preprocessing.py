from PIL import Image
import os 
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
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

    for name, feature_func in list_of_feature.items():
        print(f"{animal} - {name}")
        
        df = pd.DataFrame()
        nr = 0

        for file in tqdm(os.listdir(folder)):
            full_path = os.path.join(folder, file)

            if not file.endswith(".jpg"):
                continue

            if not is_valid_image(full_path):
                continue

            img = cv2.imread(full_path)
            if img is None:
                continue

            if img.shape[0] < 300 or img.shape[1] < 280:
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
        # after for file loop ends
        if not df.empty:
            df.to_csv(
                f"data/{animal}_{name}.csv",
                mode="a",
                header=not os.path.exists(f"data/{animal}_{name}.csv"),
                index=False
            )


def run_pipeline(animals = ["Dog","Cat"],
    path = "data/PetImages/",list_of_feature = None):

    for ani in animals:
        print(ani)
        img_to_csv_data(animal= ani,path=path,list_of_feature = list_of_feature)

        for name in list_of_feature:
            df = pd.read_csv(f"data/{ani}_{name}.csv")
       
            num_features = df.shape[1] - 1  # last column = label

            columns =  ["animal"] + [f"f_{i}" for i in range(num_features)] 

            df.columns = columns

            df.to_csv(f"data/{ani}_{name}.csv", index=False)


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
if __name__ == "__main__":
    list_of_feature_methods = {"HOG" : extract_HOG,"LBP":extract_lbp }
    path_list = [
    ["data/Cat_HOG.csv", "data/Dog_HOG.csv", "data/shuffled_HOG.csv"],
    ["data/Cat_LBP.csv", "data/Dog_LBP.csv", "data/shuffled_LBP.csv"]]
    run_pipeline(list_of_feature=list_of_feature_methods)
    for path in path_list:
        interleave_csv(cat_path=path[0],dog_path=path[1],output_path=path[2],chunksize=5000)
    #import setup_main 
    