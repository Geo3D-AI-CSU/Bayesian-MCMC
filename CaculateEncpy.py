import pandas as pd
import os
import numpy as np
from multiprocessing import  Process
from gempy.bayesian.fields import compute_prob, calculate_ie_masked

#获得指定路径下文件指定n列的array
#输入：path（string）路径
#      colum_lis（list）需要用到的列
#输出：array_model（np.array）指定n列的array
def GetArray(path,colum_list):
    df_model=pd.read_csv(path,usecols=colum_list)
    array_model=df_model.values
    return array_model

#获得交叉熵csv
#输入：folder_path（string）所有模型所在路径
#      save_name（list）保存的文件以及路径
def GetEntropyBlock(folder_path,save_name):
    lith_blocks = np.array() 
    model_list=os.listdir(folder_path)
    #第四列为地层属性值
    for name in model_list:
        var_block=GetArray(folder_path+'\\'+name,[4])
        lith_blocks=np.append(lith_blocks,var_block)
        print( path+' finished')
    lith_blocks = lith_blocks.reshape(len(model_list), -1)
    prob_block = compute_prob(lith_blocks)
    entropy_block = calculate_ie_masked(prob_block)
    path=folder_path+'\\'+model_list[0]
    model_grid=GetArray(path,[0,1,2])
    #沿着矩阵的第二个轴拼接，接上坐标值【012】列
    entropy_grid=np.c_[model_grid,entropy_block]
    np.savetxt(save_name, entropy_grid, delimiter=",")
    return True

# def GetM100(path,colum_list,name):
#     df_model=pd.read_csv(path,header=None)
#     df_model[colum_list]=df_model[colum_list]*10
#     df_model.to_excel(name+'X10.xlsx',index=None)
#     df_model[colum_list]=df_model[colum_list]*10
#     df_model.to_excel(name+'X100.xlsx',index=None)
      

if __name__ =="__main__":
    # GetEntropyBlock('models_after','Entropy_after.csv')
    procs = []
    procs.append(Process(target= GetEntropyBlock,args=('models_after1216','Entropy_after1218.csv')))
    procs.append(Process(target= GetEntropyBlock,args=('models_before1216','Entropy_before1218.csv')))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()



