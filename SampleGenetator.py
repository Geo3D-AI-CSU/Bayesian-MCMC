import os
import multiprocessing
import copy
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from IPython.core.pylabtools import figsize


##############################约束数据计算##############################

#获得一组三维点集趋势面法向量
#输入：points_set（np.array(n,3)）面上的采样点坐标
#输出：nVector（list(3)）:趋势面法向量[x,y,z]
def GetTendSurface(points_set):
    min_x=min(points_set[:,0])
    min_y=min(points_set[:,1])
    min_z=min(points_set[:,2])
    points_set[:,0]=points_set[:,0]-min_x
    points_set[:,1]=points_set[:,1]-min_y
    points_set[:,2]=points_set[:,2]-min_z
    tmp_A=copy.deepcopy(points_set)
    tmp_A[:,2]=1
    tmp_b=copy.deepcopy(points_set[:,2])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit=(A.T * A).I* A.T * b
    nVector=[float(fit[0]),float(fit[1]),-1]
   
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111, projection='3d')
    # ax1.set_xlabel("x")
    # ax1.set_ylabel("y")
    # ax1.set_zlabel("z")
    # ax1.scatter(points_set[:,0],points_set[:,1],points_set[:,2],c='r',marker='o')
    # x_p = np.linspace(0,6050,50)
    # y_p = np.linspace(0,3000,50)
    # # x_p = np.linspace(min(points_set[:,0]),max(points_set[:,0]),50)
    # # y_p = np.linspace(min(points_set[:,1]),max(points_set[:,1]),50)
    # x_p, y_p = np.meshgrid(x_p, y_p)
    # z_p = a * x_p + b * y_p +c
    # #ax1.quiver(0,0,0,-10*X[0,0],-10*X[1,0],10)
    # ax1.plot_wireframe(x_p, y_p, z_p, rstride=50, cstride=50)
    # plt.show()

    return nVector

#获得三维点集趋势面的倾角
#输入：points_set（np.array(n,3)）面上的采样点坐标
#输出：dip_angel（dtype.float32）趋势面倾角
def GetDipAngel(points_set): 
    gradient_vector=GetTendSurface(points_set)
    z_vector=np.array([0,0,1])
    strike_vector=np.cross(z_vector,gradient_vector)
    dip_vector=np.cross(strike_vector,gradient_vector)
    dip_angel=np.arctan2(abs(dip_vector[2]),math.sqrt(dip_vector[0]**2+dip_vector[1]**2))
    dip_angel=np.rad2deg(dip_angel)
    dip_angel=(dip_angel+360)%360
    return dip_angel

#获得三维点集趋势面的走向
#输入：points_set（np.array(n,3)）面上的采样点坐标
#输出：strike_angel（dtype.float32）趋势面倾角
def GetStrikeAngel(points_set):
    gradient_vector=GetTendSurface(points_set)
    z_vector=np.array([0,0,1])
    strike_vector=np.cross(z_vector,gradient_vector)
    strike_angel=np.arctan2(abs(strike_vector[0]),abs(strike_vector[1]))
    strike_angel=np.rad2deg(strike_angel)
    strike_angel=(strike_angel+360)%360
    return strike_angel

#################################采样过程#################################
import pymc as pm
from pymc.Matplot import plot

#获得两组三维点集的最短平均距离
#输入：points_set_a（np.array(n,3)）采样点坐标
#      points_set_b（np.array(n,3)）采样点坐标
#输出：min_mean_dis（dtype.float32）最短平均距离
def GetMinMeanDis(points_set_a,points_set_b):
    dis_matrix=distance.cdist(points_set_a, points_set_b, 'euclidean')
    min_dis=np.min(dis_matrix,axis=1)
    min_mean_dis=np.mean(min_dis)
    return min_mean_dis

# 获得一组数据中每一个数据的“col_name”列属性的正态分布
# 输入：dataframe（pd.dataframe）采样点以及属性
#      name(string) 自定义的属性别名前缀
#      col_name（string）属性列的列名 'X' 'Y'
#      tau(float32) 方差值
# 输出：distribut_set（list of pymc object）每一个点的分布集合
#      name_set(list of string) 每个分布的引用名
def GetDistributSet(dataframe,name,col_name,tau):
    distribut_set=[]
    names_set=[]
    for  index,row_i in dataframe.iterrows():
        para_name=name+str(index)
        distribut_set.append(pm.Normal(para_name,mu=row_i[col_name],tau=tau))
        names_set.append(para_name)
    return distribut_set, names_set

# 从一组分布列表中逐一采样，获得采样点
# 输入：distribut_names_set（list）分布的别名与列表中分布一一对应
#       runner_set（list）采样的列表
# 输出：samples_points（list）每一个分布采样到的点集
def GetSampleSet(distribut_names_set,runner_set):
    samples_points=[]
    # A=distribut_set.logp(为获得后验概率分布修改过库中源文件，如果新环境执行这句会报错)
    for name in distribut_names_set:
        samples_points.append(runner_set.trace(name)[:])
    return samples_points

# 封装的脚本,可进一步修改封装：从文件读取一组数据，对指定列“col_name”进行采样，保存至model_samples_before
# 输入：path（list）分布的别名与列表中分布一一对应
#      col_name（string）属性列的列名 'X' 'Y'
#      tau(float32) 方差值
#      iter(float32) 采样数量
#      burn(float32) 舍弃前多少个
#      model_num(float32) 保存多少组采样后的数据
# 输出：bool 是否完整执行
def GetNumSamples(path,col_name,tau,iter=10000,burn=5000,model_num=100):
    file=pd.read_excel(path)
    folder_name='model_data_before'
    file_name=path.split('\\')[-1]
    distribut_set,distribut_name_set=GetDistributSet(file,'distribute',col_name,tau)
    # model = pm.Model(distribut_set)
    runner = pm.MCMC(distribut_set)
    #record=np.linspace(2.5,97.5, 190)
    runner.sample(iter,burn)
    # runner.summary()
    #runner.stats()
    #runner.write_csv((file_name+'_summary'),alpha=0.05,quantiles=record)
    samples_points=GetSampleSet(distribut_name_set,runner)
    samples_points_narray=np.asarray(samples_points)
    for i in range(model_num):
        file[col_name]=samples_points_narray[:,(-1-i)]
        file_path='model_samples_before\\'+folder_name+str(iter-i)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        file.to_excel(file_path+'\\'+file_name,index=0)
    return True

#################################后验优化#################################

# 封装的脚本,便于多线程计算，可进一步修改封装：断层南部优化后采样
# 输入：
# 输出：
def getLithSouthAfter():
    path_list=['.\\Data\\SouthDicengPoint.xls']
    for path in path_list:
        
        col_name='Z'
        tau=0.0025
        iter=10000
        burn=5000
        model_num=100
        file=pd.read_excel(path)
        folder_name='model_data_after'
        file_name=path.split('\\')[-1]
        distribut_set,distribut_name_set=GetDistributSet(file,'distribute',col_name,tau)
        points_p1m_p1q=file[file.V==(-8214)].index.values
        points_p1q_c3=file[file.V==(-8429)].index.values
        #先验与后验
        #计算最小平均距离作为厚度 函数别名
        @pm.deterministic
        def p1qThick(
            points_index1=points_p1m_p1q,
            points_index2= points_p1q_c3,
            samples_lith=distribut_set,
            file=file.values):

            lith_points_1=np.zeros((len(points_index1),3))
            lith_points_2=np.zeros((len(points_index2),3))
            for i in points_index1:
                lith_points_1[1,:]=file[i,0:3]
                lith_points_1[1,2]=samples_lith[i]
            for i in points_index2:
                lith_points_2[1,:]=file[i,0:3]
                lith_points_2[1,2]=samples_lith[i]
            thick=GetMinMeanDis(lith_points_1,lith_points_2)
            return thick
        
        #后验分布
        @pm.stochastic
        def p1qThickLike(value=150,around_thick=p1qThick,mu=220,tau=1/(30*30)):
            return pm.normal_like(around_thick,mu,tau)

        like_func=[
            p1qThick,
            p1qThickLike]
        #后验采样
        runner = pm.MCMC(distribut_set+like_func)
        runner.sample(iter,burn)
        samples_points=GetSampleSet(distribut_name_set,runner)
        samples_points_narray=np.asarray(samples_points)
        for i in range(model_num):
            file[col_name]=samples_points_narray[:,(-1-i)]
            file_path='model_samples_after\\'+folder_name+str(iter-i)
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            file.to_excel( file_path+'\\'+file_name,index=0)
        #plot(runner)            

# 封装的脚本,便于多线程计算，可进一步修改封装：断层南北优化后采样
# 输入：
# 输出：
def getLithNorthAfter():
    path_list=['.\\Data\\NorthDicengPoint.xls']
    for path in path_list:
        col_name='Z'
        tau=0.0025
        iter=10000
        burn=5000
        model_num=100
        file=pd.read_excel(path)
        folder_name='model_data_after'
        file_name=path.split('\\')[-1]
        distribut_set,distribut_name_set=getDistributSet(file,'distribute',col_name,tau)

        points_start_t2b1=file[file.V==(-3734)].index.values
        points_t2b1_t1b=file[file.V==(-4872)].index.values
        points_t1b_t1m=file[file.V==(-6772)].index.values
        points_t1m_p2=file[file.V==(-7233)].index.values
        points_p1m_p1q=file[file.V==(-8214)].index.values
        points_p1q_c3=file[file.V==(-8429)].index.values

        @pm.deterministic
        def p1qThick(
            points_index1=points_p1m_p1q,
            points_index2=points_p1q_c3,
            samples_lith=distribut_set,
            file=file.values):
            lith_points_1=np.zeros((len(points_index1),3))
            lith_points_2=np.zeros((len(points_index2),3))
            for i in points_index1:
                lith_points_1[1,:]=file[i,0:3]
                lith_points_1[1,2]=samples_lith[i]
            for i in points_index2:
                lith_points_2[1,:]=file[i,0:3]
                lith_points_2[1,2]=samples_lith[i]
            thick=GetMinMeanDis(lith_points_1,lith_points_2)
            return thick
        @pm.stochastic
        def p1qThickkLike(value=140,around_thick=p1qThick,mu=250,tau=1/(30*30)):
            return pm.lognormal_like(around_thick,mu,tau)
      
        @pm.deterministic
        #到7233
        def p1mThick(
            points_index1=points_t1m_p2,
            points_index2=points_p1m_p1q,
            samples_lith=distribut_set,
            file=file.values):
            lith_points_1=np.zeros((len(points_index1),3))
            lith_points_2=np.zeros((len(points_index2),3))
            for i in points_index1:
                lith_points_1[1,:]=file[i,0:3]
                lith_points_1[1,2]=samples_lith[i]
            for i in points_index2:
                lith_points_2[1,:]=file[i,0:3]
                lith_points_2[1,2]=samples_lith[i]
            thick=GetMinMeanDis(lith_points_1,lith_points_2)
            return thick
        @pm.stochastic
        def p1mThickLike(value=150,around_thick=p1mThick,mu=600,tau=1/(50*50)):
            return pm.normal_like(around_thick,mu,tau)
        
        @pm.deterministic
        def t1mThick(
            points_index1=points_t1b_t1m,
            points_index2=points_t1m_p2,
            samples_lith=distribut_set,
            file=file.values):

            lith_points_1=np.zeros((len(points_index1),3))
            lith_points_2=np.zeros((len(points_index2),3))
            for i in points_index1:
                lith_points_1[1,:]=file[i,0:3]
                lith_points_1[1,2]=samples_lith[i]
            for i in points_index2:
                lith_points_2[1,:]=file[i,0:3]
                lith_points_2[1,2]=samples_lith[i]
            thick=GetMinMeanDis(lith_points_1,lith_points_2)
            return thick
        @pm.stochastic
        def  t1mThickLike(value=297,around_thick= t1mThick,mu=250,tau=1/(30*30)):
            return pm.normal_like(around_thick,mu,tau)


        @pm.deterministic
        def t1bThick(
            points_index1=points_t2b1_t1b,
            points_index2=points_t1b_t1m,
            samples_lith=distribut_set,
            file=file.values):
            
            lith_points_1=np.zeros((len(points_index1),3))
            lith_points_2=np.zeros((len(points_index2),3))
            for i in points_index1:
                lith_points_1[1,:]=file[i,0:3]
                lith_points_1[1,2]=samples_lith[i]
            for i in points_index2:
                lith_points_2[1,:]=file[i,0:3]
                lith_points_2[1,2]=samples_lith[i]
            thick=GetMinMeanDis(lith_points_1,lith_points_2)
            return thick
        @pm.stochastic
        def  t1bThickLike(value=297,around_thick= t1mThick,mu=1000,tau=1/(30*30)):
            return pm.normal_like(around_thick,mu,tau)
    
        @pm.deterministic
        def t2b1Thick(
            points_index1=points_start_t2b1,
            points_index2=points_t2b1_t1b,
            samples_lith=distribut_set,
            file=file.values):
            lith_points_1=np.zeros((len(points_index1),3))
            lith_points_2=np.zeros((len(points_index2),3))
            for i in points_index1:
                lith_points_1[1,:]=file[i,0:3]
                lith_points_1[1,2]=samples_lith[i]
            for i in points_index2:
                lith_points_2[1,:]=file[i,0:3]
                lith_points_2[1,2]=samples_lith[i]
            thick=GetMinMeanDis(lith_points_1,lith_points_2)
            return thick
        @pm.stochastic
        def t2b1ThickLike(value=217,around_thick=t2b1Thick,mu=650,tau=1/(45*45)):
            return pm.normal_like(around_thick,mu,tau)

        like_func=[
            p1mThick,
            p1qThick,
            t1mThick,
            t1bThick,
            t2b1Thick,
            p1mThickLike,
            p1qThickkLike,
            t1mThickLike,
            t1bThickLike,
            t2b1ThickLike,]
        runner = pm.MCMC(distribut_set+like_func)
        runner.sample(iter,burn)
        samples_points=getSampleSet(distribut_name_set,runner)
        samples_points_narray=np.asarray(samples_points)
        #看RUNNER对象追踪SAMPALE，能不能访问
        
        for i in range(model_num):
            file[col_name]=samples_points_narray[:,(-1-i)]
            file_path='model_samples_after\\'+folder_name+str(iter-i)
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            file.to_excel( file_path+'\\'+file_name,index=0)
        plot(runner)

# 封装的脚本,便于多线程计算，可进一步修改封装：断层优化后采样
# 输入：
# 输出：
def getFaultAfter():
    tau=4e-2
    col='X'
    iter=10000
    burn=5000
    model_num=100
    path='.\\Data\\FaultChange1.xls'
    file=pd.read_excel(path)
    folder_name='model_data_after'
    file_name=path.split('\\')[-1]
    distribut_set,distribut_name_set=getDistributSet(file,'distribute',col,tau)
    @pm.deterministic
    def dipAngel(params_fault=distribut_set,fault_array=file.values):              
        dip_angel = GetDipAngel(np.c_[params_fault,fault_array[:,1:3]])
        return dip_angel
    @pm.stochastic
    def dipAngelLike(value =3, dip1_angel_points=dipAngel):             
        return pm.normal_like(dip1_angel_points,mu=50,tau=1/(3*3))
    
    @pm.deterministic
    def strikeAngel(params_fault=distribut_set,fault_array=file.values):              
        dip_angel = GetStrikeAngel(np.c_[params_fault,fault_array[:,1:3]])
        return dip_angel
    @pm.stochastic
    def strikeAngelLike(value =3, dip1_angel_points=strikeAngel):             
        return pm.normal_like(dip1_angel_points,mu=70,tau=1/(3*3))

    like_func=[dipAngel,strikeAngel,dipAngelLike,strikeAngelLike]
    #model = pm.Model(distribut_set+like_func)
    #runner = pm.MCMC(model)
    runner = pm.MCMC(distribut_set+like_func)
    runner.sample(iter,burn)
    samples_points=getSampleSet(distribut_name_set,runner)
    samples_points_narray=np.asarray(samples_points)
    for i in range(model_num):
        file[col]=samples_points_narray[:,(-1-i)]
        file_path='model_samples_after\\'+folder_name+str(iter-i)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        file.to_excel(file_path+'\\'+file_name,index=0)
    plot(runner)
#################################统计值验证测试#################################
import matplotlib.pyplot as plt
figsize(15,15)
#进行统计值出图 个地层最小平均距离
def testStates():
    path ='贝叶斯\\SouthDicengPoint.xls'
    file=pd.read_excel(path)
    points_p1m_p1q=file[file.V==(-8214)].values[:,0:3]
    points_p1q_c3=file[file.V==(-8429)].values[:,0:3]
    points_c1_d3=file[file.V==(-10602)].values[:,0:3]
    points_d3_d2d=file[file.V==(-11142)].values[:,0:3]
    thick_p1q=GetMinMeanDis(points_p1m_p1q,points_p1q_c3)
    thick_d3=GetMinMeanDis(points_c1_d3,points_d3_d2d)
    fig = plt.figure(1)
    plt.hist(thick_p1q)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("South_p1q")
    plt.show()  
    plt.savefig("South_p1q.jpg")
    mean=np.mean(thick_p1q)
    var=np.var(thick_p1q)
    print("south_p1q_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(2)
    plt.hist(thick_d3)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("South_d3")
    plt.show()  
    plt.savefig("South_d3.jpg")
    mean=np.mean(thick_d3)
    var=np.var(thick_d3)
    print("south_d3_mean:"+str(mean)+"  var:"+str(var))


    path ='贝叶斯\\NorthDicengPoint.xls'
    file=pd.read_excel(path)
    points_start_t2b1=file[file.V==(-3734)].values[:,0:3]
    points_t2b1_t1b=file[file.V==(-4872)].values[:,0:3]
    points_t1b_t1m=file[file.V==(-6772)].values[:,0:3]
    points_t1m_p2=file[file.V==(-7233)].values[:,0:3]
    #points_p2_p1m=file[file.V==(-7609)].index.values
    points_p1m_p1q=file[file.V==(-8214)].values[:,0:3]
    points_p1q_c3=file[file.V==(-8429)].values[:,0:3]
    points_c3_c2=file[file.V==(-9145)].values[:,0:3]
    points_c2_c1=file[file.V==(-9674)].values[:,0:3]
    points_c1_d3=file[file.V==(-10602)].values[:,0:3]
    points_d3_d2d=file[file.V==(-11142)].values[:,0:3]
    points_d2d_d1y=file[file.V==(-11717)].values[:,0:3]
    points_d1y_d1n=file[file.V==(-12020)].values[:,0:3]
    points_d1n_end=file[file.V==(-12154)].values[:,0:3]

    thick_t2b1=GetMinMeanDis(points_start_t2b1, points_t2b1_t1b)
    thick_t1b=GetMinMeanDis(points_t2b1_t1b,points_t1b_t1m)
    thick_t1m=GetMinMeanDis(points_t1b_t1m,points_t1m_p2)
    thick_p1m=GetMinMeanDis(points_t1m_p2,points_p1m_p1q)
    thick_p1q=GetMinMeanDis(points_p1m_p1q,points_p1q_c3)
    thick_c3=GetMinMeanDis(points_p1q_c3,points_c3_c2)
    thick_c2=GetMinMeanDis(points_c3_c2,points_c2_c1)
    thick_c1=GetMinMeanDis(points_c2_c1,points_c1_d3)
    thick_d3=GetMinMeanDis(points_c1_d3,points_d3_d2d)
    thick_d2d=GetMinMeanDis(points_d3_d2d,points_d2d_d1y)
    thick_d1y=GetMinMeanDis(points_d2d_d1y,points_d1y_d1n)
    thick_d1n=GetMinMeanDis(points_d1y_d1n,points_d1n_end)

    fig = plt.figure(3)
    plt.hist(thick_t2b1)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_t2b1")
    plt.show()  
    plt.savefig("north_t2b1.jpg")
    mean=np.mean(thick_t2b1)
    var=np.var(thick_t2b1)
    print("north_t2b1_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(4)
    plt.hist(thick_t1b)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_t1b")
    plt.show()  
    plt.savefig("north_t1b.jpg")
    mean=np.mean(thick_t1b)
    var=np.var(thick_t1b)
    print("north_t1b_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(5)
    plt.hist(thick_t1m)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_t1m")
    plt.show()  
    plt.savefig("north_t1m.jpg")
    mean=np.mean(thick_t1m)
    var=np.var(thick_t1m)
    print("north_t1m_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(6)
    plt.hist(thick_p1m)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_p1m")
    plt.show()  
    plt.savefig("north_p1m.jpg")
    mean=np.mean(thick_p1m)
    var=np.var(thick_p1m)
    print("north_p1m_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(7)
    plt.hist(thick_p1q)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_p1q")
    plt.show()  
    plt.savefig("north_p1q.jpg")
    mean=np.mean(thick_p1q)
    var=np.var(thick_p1q)
    print("north_p1q_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(8)
    plt.hist(thick_c3)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_c3")
    plt.show()  
    plt.savefig("north_c3.jpg")
    mean=np.mean(thick_c3)
    var=np.var(thick_c3)
    print("north_c3_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(9)
    plt.hist(thick_c2)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_c2")
    plt.show()  
    plt.savefig("north_c2.jpg")
    mean=np.mean(thick_c2)
    var=np.var(thick_c2)
    print("north_c2_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(10)
    plt.hist(thick_c1)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_c1")
    plt.show()  
    plt.savefig("north_c1.jpg")
    mean=np.mean(thick_c1)
    var=np.var(thick_c1)
    print("north_c1_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(11)
    plt.hist(thick_d3)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_d3")
    plt.show()  
    plt.savefig("north_d3.jpg")
    mean=np.mean(thick_d3)
    var=np.var(thick_d3)
    print("north_d3_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(12)
    plt.hist(thick_d2d)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_d2d")
    plt.show()  
    plt.savefig("north_d2d.jpg")
    mean=np.mean(thick_d2d)
    var=np.var(thick_d2d)
    print("north_d2d_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(13)
    plt.hist(thick_d1y)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_d1y")
    plt.show()  
    plt.savefig("north_d1y.jpg")
    mean=np.mean(thick_d1y)
    var=np.var(thick_d1y)
    print("north_d1y_mean:"+str(mean)+"  var:"+str(var))

    fig = plt.figure(14)
    plt.hist(thick_d1n)
    plt.xlabel("num")
    plt.ylabel("mindis")
    plt.title("north_d1n")
    plt.show()  
    plt.savefig("north_d1n.jpg")
    mean=np.mean(thick_d1n)
    var=np.var(thick_d1n)
    print("north_d1n_mean:"+str(mean)+"  var:"+str(var))

#进行统计值断层趋势面法向，倾角，走向
def testFault():
    path ='.\\Data\\FaultChange2.xls'
    file=pd.read_excel(path)
    gradient_vector = GetTendSurface(file.values[:,0:3])
    dip_angel = GetDipAngel(gradient_vector)
    strike_angel = GetStrikeAngel(gradient_vector)
    print(gradient_vector)
    print(strike_angel)
    print(dip_angel)

if __name__ == "__main__":
    # testFault()
    #添加似然之前
    path_list=['.\\Data\\SouthDicengPoint.xls','.\\Data\\NorthDicengPoint.xls']
    for path in path_list:
         GetNumSamples(path,'Z',0.0025,100,50)
    GetNumSamples('Data\\FaultChange2.xls','X',0.0025,10000,5000)
    print('100 models finished')
    #添加似然之后
    procs = []
    procs.append(multiprocessing.Process(target=getLithSouthAfter))
    procs.append(multiprocessing.Process(target=getLithNorthAfter))
    procs.append(multiprocessing.Process(target=getFaultAfter))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    print('100 models finished')

