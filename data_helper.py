import pandas as pd
import numpy as np

def read_exp_csv(file_name):
    df = pd.read_csv(file_name) 
    xd = np.array(list(df.iloc[0:,0]))
    xs = np.array(list(df.iloc[0:,1]))
    f = np.array(list(df.iloc[0:,8]))
    u_xd = np.array(list(df.iloc[0:,9]))
    u_xs = np.array(list(df.iloc[0:,10]))
    u_f = np.array(list(df.iloc[0:,13]))
    return np.array([xd,xs,f,u_xd,u_xs,u_f])

def read_csv(file_name):
    df = pd.read_csv(file_name, header=None) 
    xd = np.array(list(df.iloc[0:,0]))
    xs = np.array(list(df.iloc[0:,1]))
    f = np.array(list(df.iloc[0:,2]))
    u_xd = np.array(list(df.iloc[0:,3]))
    u_xs = np.array(list(df.iloc[0:,4]))
    u_f = np.array(list(df.iloc[0:,5]))
    return np.array([xd,xs,f,u_xd,u_xs,u_f])
