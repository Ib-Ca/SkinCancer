import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, shutil
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



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
'''
metadata.set_index('image_id', inplace=True)
folder_1 = os.listdir('../ham10000_images_part_1')
folder_2 = os.listdir('../ham10000_images_part_2')
train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

for image in train_list:
    
    fname = image + '.jpg'
    label = metadata.loc[image,'dx']
    
    if fname in folder_1:
        src = os.path.join('../ham10000_images_part_1', fname)
        dst = os.path.join(train_dir, label, fname)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        src = os.path.join('../ham10000_images_part_2', fname)
        dst = os.path.join(train_dir, label, fname)
        shutil.copyfile(src, dst)

for image in val_list:
    fname = image + '.jpg'
    label = metadata.loc[image,'dx']
    if fname in folder_1:
        src = os.path.join('../ham10000_images_part_1', fname)
        dst = os.path.join(val_dir, label, fname)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        src = os.path.join('../ham10000_images_part_2', fname)
        dst = os.path.join(val_dir, label, fname)
        shutil.copyfile(src, dst)
'''        

# modelo Building-Training
train_csv_file = 'train_images_labels.csv'
val_csv_file = 'val_images_labels.csv'

if not os.path.exists(train_csv_file) or not os.path.exists(val_csv_file):
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    def load_images_from_dir(dir, data_list, labels_list):
        for label in os.listdir(dir):
            label_dir = os.path.join(dir, label)
            if os.path.isdir(label_dir):
                for image_file in os.listdir(label_dir):
                    if image_file.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(label_dir, image_file)
                        image = Image.open(image_path).convert('RGB')  
                        image = image.resize((28, 28))  
                        image_array = np.array(image).flatten() 
                        data_list.append(image_array)
                        labels_list.append(label) 
                        
    load_images_from_dir(train_dir, train_data, train_labels)
    load_images_from_dir(val_dir, val_data, val_labels)

    df_train = pd.DataFrame(train_data)
    df_train['label'] = train_labels
    df_val = pd.DataFrame(val_data)
    df_val['label'] = val_labels
    df_train.to_csv(train_csv_file, index=False)
    df_val.to_csv(val_csv_file, index=False)
    
    print(f"Archivos CSV creados: {train_csv_file}, {val_csv_file}")
else:
    print("Los archivos CSV ya existen. Cargando datos de los archivos existentes.")

if  os.path.exists(train_csv_file) and  os.path.exists(val_csv_file):
    train_data = pd.read_csv('train_images_labels.csv')
    val_data = pd.read_csv('val_images_labels.csv')
    #entrenamiento
    X_train = train_data.drop('label', axis=1)  
    y_train = train_data['label']  
    #validamiento
    X_val = val_data.drop('label', axis=1)  
    y_val = val_data['label'] 
    #transformar a num
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    print("fin de carga de datos")
#entrenamiento
dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
dval = xgb.DMatrix(X_val, label=y_val_encoded)
params = {
    'objective': 'multi:softmax', 
    'num_class': 7,   
    'max_depth': 6,                  
    'eta': 0.3,  
    'min_child_weight': 1,    
    'subsample': 0.3,
    'colsample_bytree': 0.3,
    'eval_metric': 'mlogloss',
    'seed': 77                                  
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,  
    evals=[(dval, 'validation')],
    early_stopping_rounds=50  
)

#prediccion
y_pred = model.predict(dval)
print("Accuracy:", accuracy_score(y_val_encoded, y_pred))
print(classification_report(y_val_encoded, y_pred))

xgb.plot_importance(model)
plt.show()

end_time = time.time()
execution_time = end_time - start_time
#tiempo de ejecucion
print(f"Tiempo de ejecución: {execution_time:.4f} segundos")