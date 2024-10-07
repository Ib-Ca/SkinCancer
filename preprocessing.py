import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, shutil
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


start_time = time.time()
#archivos
data_path = os.path.join("..")
metadata = pd.read_csv(os.path.join(data_path, "HAM10000_metadata.csv"))
hmnist_8_8_l = pd.read_csv(os.path.join(data_path, "hmnist_8_8_L.csv"))
hmnist_8_8_rgb = pd.read_csv(os.path.join(data_path, "hmnist_8_8_RGB.csv"))
hmnist_28_28_l = pd.read_csv(os.path.join(data_path, "hmnist_28_28_L.csv"))
hmnist_28_28_rgb = pd.read_csv(os.path.join(data_path, "hmnist_28_28_RGB.csv"))
base_dir = 'base_dir'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')


#cuantas imagenes estan asociadas a cada lesion id
df = metadata.groupby('lesion_id').count()
#filtrar las lesion_id con una sola imagen
df = df[df['image_id'] == 1]
df.reset_index(inplace=True)
#print(df.head())

#identificar lesion_id con duplicados y no duplicados
lesion_counts = metadata['lesion_id'].value_counts()
metadata['duplicates'] = metadata['lesion_id'].map(lambda x: 'no_duplicates' if lesion_counts[x] == 1 else 'has_duplicates')
#print(metadata.head())
#print(metadata['duplicates'].value_counts())

#separar duplicado y no dup
df = metadata[metadata['duplicates'] == 'no_duplicates']
#print(df.shape)

#crear df validacion con lo no duplicado
y = df['dx']
_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)
#print(df_val.shape)
#print(df_val['dx'].value_counts())

#seperar valores para validacion y entrenamiento
def identify_val_rows(x):
    val_list = list(df_val['image_id'])
    
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'
metadata['train_or_val'] = metadata['image_id']
metadata['train_or_val'] = metadata['train_or_val'].apply(identify_val_rows)
df_train = metadata[metadata['train_or_val'] == 'train']
#print(len(df_train))
#print(len(df_val))
#print("valores de entrenamiento: ", df_train['dx'].value_counts())
#print("valores de validacion: ", df_val['dx'].value_counts())

#mover archivos
metadata.set_index('image_id', inplace=True)
folder_1 = os.listdir('../ham10000_images_part_1')
folder_2 = os.listdir('../ham10000_images_part_2')
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

def is_directory_empty(dir_path):
    return len(os.listdir(dir_path)) == 0

if is_directory_empty(train_dir) and is_directory_empty(val_dir):
    metadata.set_index('image_id', inplace=True)
    folder_1 = os.listdir('../ham10000_images_part_1')
    folder_2 = os.listdir('../ham10000_images_part_2')
    train_list = list(df_train['image_id'])
    val_list = list(df_val['image_id'])
    for image in train_list:
        fname = image + '.jpg'
        label = metadata.loc[image, 'dx']
        if fname in folder_1:
            src = os.path.join('../ham10000_images_part_1', fname)
            dst = os.path.join(train_dir, label, fname)
            shutil.copyfile(src, dst)
        elif fname in folder_2:
            src = os.path.join('../ham10000_images_part_2', fname)
            dst = os.path.join(train_dir, label, fname)
            shutil.copyfile(src, dst)
    for image in val_list:
        fname = image + '.jpg'
        label = metadata.loc[image, 'dx']
        if fname in folder_1:
            # Copiar desde part_1
            src = os.path.join('../ham10000_images_part_1', fname)
            dst = os.path.join(val_dir, label, fname)
            shutil.copyfile(src, dst)
        elif fname in folder_2:
            # Copiar desde part_2
            src = os.path.join('../ham10000_images_part_2', fname)
            shutil.copyfile(src, dst)
else:
    print("Los archivos ya han sido copiados previamente.")


end_time = time.time()
execution_time = end_time - start_time
#tiempo de ejecucion
print(f"Tiempo de ejecución: {execution_time:.4f} segundos")