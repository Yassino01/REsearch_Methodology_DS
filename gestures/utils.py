from time import time
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
sns.set();
from sklearn.metrics import accuracy_score


RANDOM_SEED = 42

class Timer:
    def __init__(self):
        self.time_start = None
        self.cache = []
    def tik(self):
        self.start = time()
    def tak(self):
        duration =  time() - self.start 
        self.cache.append(duration)
        return duration
    def mean_duration(self):
        return np.mean(self.cache)
    def reset():
        self.cache = []



def get_data(addr = "data/SG24_dataset.h5"):
    data = h5py.File(addr)
    df_dict = {"user" : data['User'][:, :].flatten(), "target" : data['Target'][:, :].flatten()}
    for i in range(data['Predictors'].shape[0]):
        df_dict["sensor_{}".format(i)] = data['Predictors'][i, :]

    df = pd.DataFrame( df_dict )
    return df

def split(data, n_user_test = 2,seed = 42):
    np.random.seed(seed)
    df = data.copy()

    unique_user = df['user'].unique().astype(int)
    np.random.shuffle(unique_user)
    test_user = unique_user[:n_user_test]
    train_user = unique_user[n_user_test:]

    mask_train = df['user'].isin(train_user)
    mask_test = df['user'].isin(test_user)
    return df[mask_train], df[mask_test]



def oversample(df):
  unique_user = df['user'].unique()
  nb_obs_per_user = df['user'].value_counts()
  max_obs = nb_obs_per_user.max()

  to_add = pd.DataFrame()

  for user in unique_user:
    nb_obs = nb_obs_per_user[user]
    nb_obs_to_add = max_obs - nb_obs

    if not nb_obs_to_add > 0:
      continue

    for _ in range(nb_obs_to_add // nb_obs):
        tmp = df[df['user']==user]
        to_add = pd.concat( (to_add, tmp) )

    tmp = df[df['user']==user].sample(nb_obs_to_add % nb_obs, replace=True)
    to_add = pd.concat((to_add, tmp), axis=0)

  return pd.concat((to_add, df), axis=0).sample(frac=1.)



def normalise(X1, X2):
    mu = X1.mean(axis=0, keepdims=True)
    sigma = X1.std(axis=0, keepdims=True)

    return (X1 - mu) / sigma, (X2 - mu) / sigma 



### cross val  train -> train / val

def get_cross_val_data(data, n_user_per_fold=2, oversample_train = False, seed=42):
  np.random.seed(seed)
  df = data.copy()
  n_user = df['user'].nunique()
  unique_user = df['user'].unique()
  np.random.shuffle(unique_user)
  
  for i in range(0, n_user, n_user_per_fold):
    val_user = unique_user[i:i+n_user_per_fold]
    
    val_mask = data['user'].isin(val_user)
    train_df = df[~val_mask].sample(frac=1.)
    val_df = oversample(df[val_mask])

    if oversample_train:
      train_df = oversample(train_df)

    

    train_x = train_df.values[:, 2:]
    train_y = train_df.values[:, 1]
    val_x = val_df.values[:, 2:]
    val_y = val_df.values[:, 1]

    train_x , val_x = normalise(train_x,val_x)
    yield train_x, train_y, val_x, val_y




def pool(modelDict, df, n_user_per_fold=1, oversample_train=False, random_seed = RANDOM_SEED, transforms=None):
    train_df = df.copy()
    accuracy_stats = dict()
    time_stats = dict()

    for name, model in modelDict.items():
        timer_pred = Timer()
        timer_train = Timer()
        #timer.tik()

        val_acc_list = []
        train_acc_list = []
        time_pred = []

        #print(f"Model {name}")
        data_iter = get_cross_val_data(train_df, n_user_per_fold=1, oversample_train=oversample_train, seed = random_seed)
        for x_train, y_train, x_val, y_val in data_iter:
            if transforms is not None:
                x_train, x_val = transforms(x_train, x_val)

            timer_train.tik()
            model.fit(x_train, y_train)
            timer_train.tak()
            
            
            y_train_pred = model.predict(x_train)
            
            timer_pred.tik()
            y_val_pred = model.predict(x_val)
            timer_pred.tak()

            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            #print(train_acc, val_acc)
        
        accuracy_stats[name] = {
            "mean_train_accuracy" : np.round(np.mean(train_acc_list), 4),
            "std_train_accuracy" : np.round(np.std(train_acc_list), 4),
            "mean_val_accuracy" : np.round(np.mean(val_acc_list), 4),
            "std_val_accuracy" : np.round(np.std(val_acc_list), 4),
            "training_time" : np.round(timer_train.mean_duration(), 2),
            "prediction_time" : np.round(timer_pred.mean_duration(), 2)
        }

        time_stats[name] = {
            "training_time" : np.round(timer_train.mean_duration(), 2),
            "prediction_time" : np.round(timer_pred.mean_duration(), 2)
        }
    return pd.DataFrame(accuracy_stats).T.sort_values(by="mean_val_accuracy", ascending=False).T, pd.DataFrame(time_stats)

