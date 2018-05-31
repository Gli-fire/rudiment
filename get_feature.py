from cub_util import CUB200
import pandas as pd
import os
from keras.applications import VGG19, InceptionV3, Xception, ResNet50
from keras.applications.imagenet_utils import preprocess_input as preprocess_type1
import numpy as np

CUB_DIR = 'C:\\Users\\wfr\\Desktop\\files\\AI\\PROJECT\\data'
FEATURES_DIR = 'FEATURES_DIR'
if not os.path.exists(FEATURES_DIR ):
    os.makedirs(FEATURES_DIR)
    
def load_features_compute_once(model, im_size, preprocess, save_path):
        
    if os.path.exists(save_path):
        data = pd.read_csv(save_path, compression='gzip', header=0, index_col=0)
        X = data.values 
        y = data.index.values
    else:
        X, y = CUB200(CUB_DIR, image_size=im_size).load_dataset()
        X = np.array(X).astype(float)
        print('Begin train:')
        X = model(include_top=False, weights="imagenet", pooling='avg').predict(preprocess(X))
        print('End train:')
        pd.DataFrame(X, index=y).to_csv(save_path, compression='gzip', header=True, index=True)

    return X, y

#分别尝试原数据，dlib后数据，人工筛选后数据
X_resnet, y_resnet = load_features_compute_once(ResNet50, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_resnet"))
#size=(2254,2048)
X_resnet, y_resnet = load_features_compute_once(ResNet50, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_resnet_dlib"))
#size=(1132,2048)
X_resnet, y_resnet = load_features_compute_once(ResNet50, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_resnet_new"))
#size=(1842,2048)


class_mapping = {'1.brother-sister':0,
                 '2.father-daughter':0,
                 '3.mother-son':0,
                 '4.partner':1,
                 '5.lover':1,
                 '6.spouse':1}

Y = pd.DataFrame(y_resnet)[0].map(class_mapping)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resnet, Y, test_size=0.3, 
    random_state=0, stratify=Y)

#使用Logistic回归
from sklearn.linear_model import LogisticRegression
#进行网格搜索参数优化
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
#进行数据标准化
pipe_lr = make_pipeline(StandardScaler(),
                         LogisticRegression(penalty='l2',random_state=1))   
#选择几个可能的参数
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'logisticregression__C': param_range}]

gs = GridSearchCV(estimator=pipe_lr, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print('Train accuracy: %.3f' % gs.best_score_)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
clf.get_params() 

#使用随机森林
from sklearn.ensemble import RandomForestClassifier                                 
#进行网格搜索参数优化
pipe_lr = RandomForestClassifier(max_depth=None,min_samples_split=2, random_state=0)
param_range = range(10, 100,5)
param_grid = [{'n_estimators': param_range}]

gs = GridSearchCV(estimator=pipe_lr, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print('Train accuracy: %.3f' % gs.best_score_)
clf = gs.best_estimator_
clf.fit(X_train, y_train)
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
clf.get_params() 
