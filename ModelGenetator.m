addpath ".\AdaHRBF"
attitude_north = xlsread('Data\NFattitude.xlsx');
attitude_south = xlsread('Data\SFattitude.xlsx');
attitude_north=AttitudeToVector(attitude_north);
attitude_south=AttitudeToVector(attitude_south);
gradient_north=DipvectorToGradientvector(attitude_north);
gradient_south=DipvectorToGradientvector(attitude_south);
k=1e-4;
basefunctype=1;

maindir = 'data\';
subdir  = dir( maindir );
% data文件夹下所有采样后的采样点目录，逐一插值计算模型
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' )||...
        isequal( subdir( i ).name, '..')||...
        ~subdir( i ).isdir)               % 如果不是目录则跳过
        continue;
    end
%   子文件目录
    subdirpath = fullfile( maindir, subdir( i ).name);
%   每个模型文件夹下有断层、南部、北部三个文件表示三个采样数据
    dat = dir( subdirpath );             
    fault_path = fullfile( maindir, subdir( i ).name, dat( 3 ).name);
    north_path = fullfile( maindir, subdir( i ).name, dat( 4 ).name); 
    south_path = fullfile( maindir, subdir( i ).name, dat( 5 ).name);
    attributes_f = xlsread(fault_path);
    attributes_n = xlsread(north_path);
    attributes_s = xlsread(south_path);
    
%   HRBF插值过程
    attributes_north = attributes_n;
    attributes_south = attributes_s;
    attributes_fault = attributes_f;

    % geting the scope of study area and initial the mesh grid
     [meshgrid_NX,meshgrid_NY,meshgrid_NZ] = meshgrid(666500:50:673900,2570150:50:2576450,0:25:1150);
     [meshgrid_SX,meshgrid_SY,meshgrid_SZ] = meshgrid(666500:50:673900,2570150:50:2576450,0:25:1150);
     
    %HRBF: interpolating an initial 3D scalar field function in northern sub-domain
    [alph_n,bravo_n,charlie_n]=GetParameters(1e-4,1,attributes_north,gradient_north);
    valueGrid_north=GetValueGrid(k,basefunctype,meshgrid_NX,meshgrid_NY,meshgrid_NZ,attributes_north,gradient_north,alph_n,bravo_n,charlie_n);

    %HRBF: interpolating an initial 3D scalar field function in southern sub-domain 
    [alphs,betas,charlies]=GetParameters(1e-4,1,attributes_south,gradient_south);
    valueGrid_south=GetValueGrid(k,basefunctype,meshgrid_SX,meshgrid_SY,meshgrid_SZ,attributes_south,gradient_south,alphs,betas,charlies);
    
    %RBF: interpolating a 3D scalar field for fault
    [alph,~,charlie]=GetParameters(1e-4,1,attributes_fault);
    valueGrid_fault=GetValueGrid(k,basefunctype,meshgrid_NX,meshgrid_NY,meshgrid_NZ,attributes_fault,nan,alph,nan,charlie);
    clear alph_n bravo_n charlie_n  alphs betas charlies
 
    index_s=find(valueGrid_fault>=1);
    index_f=find(valueGrid_fault>-1 & valueGrid_fault<1);
    valueGrid_north(index_s)=valueGrid_south(index_s);
    valueGrid_north(index_f)=0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%可视化过程################
%     value_list=[4872,7233,8429,9145,9674,10602,11717,8429,9145,9674,10602,11142];
%     color_list={'#ffe7ff','#fedfff','#ecffb0','#e2e2e2','#d0cfe1','#c5c4ca','#ffdfc0','#ecffb0','#e2e2e2','#d0cfe1','#c5c4ca','#ffe7cf'};
%     area=[664700 675100 2568400 2578100 0 1150];
%     color_axis=[4000,13000];
%     % displaying stratigraphic interfaces in the northern subdomain
%     fig2=figure('Name','HRBFisosurfaces');         
%     figure(fig2);
%     % stratigraphic interfaces with initial gradient magnitude
%     subplot(1,2,1);
%     light('position',[-0.5,-0.5,0.5],'style','local','color','[0.5,0.5,0.5]')
%     grid on; box on;
%     axis(area);
%     axis equal;
%     set(gca,'XLim',[area(1) area(2)]);
%     set(gca,'YLim',[area(3) area(4)]);
%     set(gca,'ZLim',[area(5) area(6)]);
%     for i=1:7
%         pic = patch(isosurface(meshgrid_NX,meshgrid_NY,meshgrid_NZ,valueGrid_north,value_list(i)));
%         isonormals(meshgrid_NX,meshgrid_NY,meshgrid_NZ,valueGrid_north,pic)
%         set(pic,'FaceColor',color_list{i},'EdgeColor','none');
%         hold on;
%     end
%     % dsiplaying faults interfaces
%     pic_f = patch(isosurface(meshgrid_NX,meshgrid_NY,meshgrid_NZ,valueGrid_fault,0));
%     isonormals(meshgrid_NX,meshgrid_NY,meshgrid_NZ,valueGrid_fault,pic_f)
%     set(pic_f,'FaceColor','red','EdgeColor','none');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%保存结果

    %输出
    m=size(valueGrid_north(:),1);
    %A为M行（网格元素个数）1列，初始值-1表示Nan
    A=-1.*ones(m,1);
    clear attributes_fault attributes_north attributes_south meshgrid_SX meshgrid_SY meshgrid_SZ valueGrid_south valueGrid_fault
    %valueMartix为M行（网格元素个数）5列(X,Y,Z,地层属性值，地层标号防止读写时候把其他值读为字符串，所以用数字编号
    valueMartix=[meshgrid_NX(:),meshgrid_NY(:),meshgrid_NZ(:),valueGrid_north(:),A];
    clear meshgrid_NX meshgrid_NY meshgrid_NZ valueGrid_north A
    
    index_=find(valueMartix(:,4)==0);
    valueMartix(index_,5)=0;
    index_=find(valueMartix(:,4)>1898 & valueMartix(:,4)<3734);
    valueMartix(index_,5)=1;
    index_=find(valueMartix(:,4)>3734 & valueMartix(:,4)<4872);
    valueMartix(index_,5)=2;
    index_=find(valueMartix(:,4)>4872 & valueMartix(:,4)<6772);
    valueMartix(index_,5)=3;
    index_=find(valueMartix(:,4)>6772 & valueMartix(:,4)<7233);
    valueMartix(index_,5)=4;
    index_=find(valueMartix(:,4)>7233 & valueMartix(:,4)<8214);
    valueMartix(index_,5)=5;
    index_=find(valueMartix(:,4)>8214 & valueMartix(:,4)<8429);
    valueMartix(index_,5)=6;
    index_=find(valueMartix(:,4)>8429 & valueMartix(:,4)<9145);
    valueMartix(index_,5)=7;
    index_=find(valueMartix(:,4)>9145 & valueMartix(:,4)<9674);
    valueMartix(index_,5)=8;
    index_=find(valueMartix(:,4)>9674 & valueMartix(:,4)<10602);
    valueMartix(index_,5)=9;
    index_=find(valueMartix(:,4)>10602 & valueMartix(:,4)<11142);
    valueMartix(index_,5)=10;
    index_=find(valueMartix(:,4)>11142 & valueMartix(:,4)<11717);
    valueMartix(index_,5)=11;
    index_=find(valueMartix(:,4)>11717 & valueMartix(:,4)<12020);
    valueMartix(index_,5)=12;
    index_=find(valueMartix(:,4)>12020);
    valueMartix(index_,5)=13;
    %保存每个模型
    name=strcat("model_after",num2str(i-3));
    name=strcat(name,".csv");
    path=strcat("models_after1216\",name);
    csvwrite(path,valueMartix)
    clear valueMartix index_
end
 