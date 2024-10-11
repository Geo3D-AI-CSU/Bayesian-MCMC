import pandas as  pd


#study_area=[664400.0, 678100.0, 2567500.0 ,2577200.0,-400, 900]
study_area=[0, 0, 12568000.0 ,12580700.0, -11000, 11000]
"""
截取研究区域的数据
grid_dataframe(pd.dataframe)GRID的XYZ以及属性值
study_area（list）研究区域的坐标区间
"""
def getPointsStudyArea(grid_dataframe,study_area):
    new_grid_dataframe1=grid_dataframe[
        (grid_dataframe['X']>study_area[0])
        & (grid_dataframe['X']<study_area[1])
        & (grid_dataframe['Y']>study_area[2]) 
        & (grid_dataframe['Y']<study_area[3])
        & (grid_dataframe['Z']>study_area[4])
        & (grid_dataframe['Z']<study_area[5])]
    return new_grid_dataframe1

"""
截取需要研究的地层
grid_dataframe(pd.dataframe)GRID的XYZ以及属性值
"""    
def getUsedLithPoints(points_dataframe):
    points_dataframe1=points_dataframe[
        (points_dataframe['formation']=='fault')
        |(points_dataframe['formation']=='T2b1')
        |(points_dataframe['formation']=='T1b')
        |(points_dataframe['formation']=='T1m')
        |(points_dataframe['formation']=='P1m')
        |(points_dataframe['formation']=='P1q')
        |(points_dataframe['formation']=='C3')
        |(points_dataframe['formation']=='C2')
        |(points_dataframe['formation']=='C1')
        |(points_dataframe['formation']=='D3')
        |(points_dataframe['formation']=='D2d')
        |(points_dataframe['formation']=='D1y')]
    return points_dataframe1

"""
Fault[1~29]
South[]
"""

"""
当时设想数据不够导致模型失真，所以使用HRBF论文数据，截取该研究区域数据点。后用于截取用于GOCAD显示的数据
"""    
if __name__ == "__main__":
    north_point=pd.read_excel('NorthPoint.xls').dropna(axis=0)
    south_point=pd.read_excel('SouthPoint.xls').dropna(axis=0)
    fault_point=pd.read_excel('Fault.xls').dropna(axis=0)
    orinent_point=pd.read_csv('orientation1.csv').dropna(axis=0)
    domain_list=[north_point,south_point,fault_point]
    name_list=['NorthPoints.csv','SouthPoints.csv','FaultPoints.csv']
    points=[]
    for points_dataframe,save_name in zip(domain_list,name_list):
        points_study_area=getPointsStudyArea(points_dataframe,study_area)
        points_study_area=getUsedLithPoints(points_study_area)
        part=save_name.split('Points')[0]
        points_study_area['area']=part
        points_study_area.to_csv(save_name,index=0)
        points.append(points_study_area) 
    orinent_point=getPointsStudyArea(orinent_point,[664400.0,678100.0,2567500.0,2577200.0,-400,1000])
    #orinent_point=getPointsStudyArea(orinent_point,study_area)
    orinent_point.to_csv('Orientation.csv',index=0)
    all_points=pd.merge(points[2],points[1],how='outer')
    all_points=pd.merge(all_points,points[0],how='outer')
    all_points.to_csv('Points1206.csv',index=0)


