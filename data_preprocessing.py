from PIL import Image
import os 
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import cv2
from itertools import zip_longest



def image_to_feature(df:pd.DataFrame,img, features):
    rows = []
    
    data = {"animal":[]}
    
    if "Cat" in img:
        animal =  1 #"Cat"
    else: animal = 0 #"Dog"
    imgread = cv2.imread(img)
    
    imgreadre = cv2.resize(imgread,(300,280))
    gray = cv2.cvtColor(imgreadre, cv2.COLOR_BGR2GRAY)
    data["animal"] = animal

    feature = feature(gray)

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
    
def img_to_csv_data(animal,path,loops_unitl_save = 5,list_of_feature = None):
    print("starting")
    nr = 0
    try:
        df = pd.DataFrame(pd.read_csv("data/data.csv"))
    except:
        df = pd.DataFrame()
    path = path + animal
    for file in tqdm(os.listdir(path)):
        full_path = os.path.join(path, file)
        if file.endswith(".jpg"):
            if is_valid_image(full_path) :
                if (cv2.imread(full_path)).shape[0] >= 300 and (cv2.imread(full_path)).shape[1] >= 280:
                    for feature in list_of_feature:
                        df = image_to_feature(df,full_path,feature[1])
                        if loops_unitl_save == nr:
                            df.to_csv("data/"+animal+feature[0]+"data.csv",mode="a",header=False,index=False)
                            df = pd.DataFrame()
                            nr = 0
                        nr +=1

def exec(animals = ["Dog","Cat"],
    path = "data/PetImages/",list_of_feature = None):
    for ani in animals:
        print(ani)
        img_to_csv_data(animal= ani,path=path,list_of_feature = list_of_feature)
        df = pd.read_csv("data/"+ani+"data.csv", header=None)
        num_features = df.shape[1] - 1  # last column = label
        columns =  ["animal"] + [f"f_{i}" for i in range(num_features)] 
        df.columns = columns
        df.to_csv("data/"+ani+"data.csv", index=False)


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



if __name__ == "__main__":
    list_of_feature = {"hog" : hog(
    gray,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    feature_vector=True),                }
    exec()
    interleave_csv(cat_path="data/Catdata.csv",dog_path="data/Dogdata.csv",output_path="data/shuffleddata.csv",chunksize=5000)
    import setup_main 
    