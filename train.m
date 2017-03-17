% You can change anything you want in this script.
% It is provided just for your convenience.
clear; clc; close all;
%  [X, map] = imread('./train/007_CP90_HP26_SD400_6796_4.jpg');
%  image(X);
%  rgb = ind2rgb(X,map);
%imshow(rgb);
% image = rgb2gray(image);
% imshow(image);

%img_path = './train/';
%class_num = 30;
%img_per_class = 80;
%img_num = 1086;


%folder_dir = dir(img_path);
%label_train = zeros(img_num,1);
poke_feat=[];
hp_feat=[];
cp_feat=[];
sd_feat=[];

img_path = './train/';
img_dir = dir([img_path,'*CP*']);
img_num = length(img_dir);
color_feat=zeros(img_num,9);
n1 = zeros(img_num,1);
n2 = zeros(img_num,1);
n3 = zeros(img_num,1);
n4 = zeros(img_num,1);
ID_train = zeros(img_num,1);
CP_train = zeros(img_num,1);
HP_train = zeros(img_num,1);
stardust_train = zeros(img_num,1);
%Defining the number of clusters for the K means
no_of_clusters=70;
for i = 1:img_num
    
    close all;
    
    image = imread([img_path,img_dir(i).name]);
    name = img_dir(i).name;
    ul_idx = findstr(name,'_'); 
    ID_train(i) = str2num(name(1:ul_idx(1)-1));
    CP_train(i) = str2num(name(ul_idx(1)+3:ul_idx(2)-1));
    HP_train(i) = str2num(name(ul_idx(2)+3:ul_idx(3)-1));
    stardust_train(i) = str2num(name(ul_idx(3)+3:ul_idx(4)-1));
    %image = imread([img_path,folder_dir(i+2).name,'/',img_dir(j).name]);
    img=imresize(image,[1000,600]);
    %imshow(img);
    poke_color = img(100:500,100:500,:);
    
    cpimg=imcrop(img,[1,83,596,330]);
    
    bwcp=im2bw(cpimg,0.95);
    
    
    try
        img_gray = rgb2gray(img);
        
    catch exception
        [X, map] = imread([img_path,img_dir(i).name]);
        mapsaved=map;
        %image(X);
        rgb = ind2rgb(X,map);
        poke_color=imcrop(rgb,[1,83,596,330]);
        img_gray=rgb2gray(rgb);
        continue
    end
    
%     R = poke_color(:,:,1);
%     G = poke_color(:,:,2);
%     B = poke_color(:,:,3);
%     
%     tR=graythresh(R)/255;
%     tG=graythresh(G)/255;
%     tB=graythresh(B)/255;
%     
%     R = mean(R(:))/255;
%     G = mean(G(:))/255;
%     B = mean(B(:))/255;
%     
%     
    R = poke_color(:,:,1);
    G = poke_color(:,:,2);
    B = poke_color(:,:,3);
    TR = graythresh(R);
    TG = graythresh(G);
    TB = graythresh(B);
    R = [mean(mean(R(1:floor(end/2),:))) mean(mean(R(floor(end/2)+1:end,:)))];
    G = [mean(mean(G(1:floor(end/2),:))) mean(mean(G(floor(end/2)+1:end,:)))];
    B = [mean(mean(B(1:floor(end/2),:))) mean(mean(B(floor(end/2)+1:end,:)))];
    meanColors = [R G B TR TG TB]/255;
    
    color_feat(i,:) = meanColors;
   poke=imcrop(img_gray,[1,83,596,330]);
%     cpimg=imcrop(img_gray,[1,1,598,100]);
    hpimg=imcrop(img_gray,[1,505,598,50]);
    sdimg=imcrop(img_gray,[300,780,120,60]);
%     
%     imshow(poke);
%     imshow(cpimg);
%     imshow(hpimg);
%     imshow(sdimg);
    
    
    pokepoints = detectSURFFeatures(poke);
    pointshp = detectSURFFeatures(hpimg);
    pointscp = detectSURFFeatures(bwcp);
    pointssd = detectSURFFeatures(sdimg);
        %points = detectHarrisFeatures(I)
    [pokemonfeat, idpoints] = extractFeatures(poke, pokepoints.selectStrongest(300),'SURFSize', 128);
    [cpfeat, cppoints] = extractFeatures(bwcp, pointscp.selectStrongest(100),'SURFSize', 128);
    [hpfeat, hppoints] = extractFeatures(hpimg, pointshp.selectStrongest(100),'SURFSize', 128);
    [sdfeat, sdpoints] = extractFeatures(sdimg, pointssd.selectStrongest(100),'SURFSize', 128);
    
        %[feat, validpoints] = extractFeatures(image, points.selectStrongest(300),'SURFSize', 128);
    poke_feat=[poke_feat;pokemonfeat];
    hp_feat=[hp_feat;hpfeat];
    cp_feat=[cp_feat;cpfeat];
    sd_feat=[sd_feat;sdfeat];
    
    n1(i) = size(pokemonfeat,1);
    n2(i) = size(cpfeat,1);
    n3(i) = size(hpfeat,1);
    n4(i) = size(sdfeat,1);
        %cell{(i-1)*img_per_class+j, 1} = feat;
        %feat_train((i-1)*img_per_class+j,:) = feature_extraction(img);
end
    

%Clustering algorithm
[idx1, pokecodebook] = kmeans(poke_feat, no_of_clusters,'MaxIter',1000');
[idx2, cpcodebook] = kmeans(cp_feat, no_of_clusters,'MaxIter',1000');
[idx3, hpcodebook] = kmeans(hp_feat, no_of_clusters,'MaxIter',1000');
[idx4, sdcodebook] = kmeans(sd_feat, no_of_clusters,'MaxIter',1000');

poke_train_features = [];
cp_train_features=[];
hp_train_features=[];
sd_train_features=[];

k=0;
%Generating the histogram or vector of visual words for each image
for i=1:img_num
    h = hist(idx1(k+1:k+n1(i)), 1:no_of_clusters);
    poke_train_features = [poke_train_features; h];
    
    k=k+n1(i);
end
k=0;
for i=1:img_num
    h = hist(idx2(k+1:k+n2(i)), 1:no_of_clusters);
    cp_train_features = [cp_train_features; h];
    k=k+n2(i);
end
k=0;
for i=1:img_num
    h = hist(idx3(k+1:k+n3(i)), 1:no_of_clusters);
    hp_train_features = [hp_train_features; h];
    k=k+n3(i);
end
k=0
for i=1:img_num
    h = hist(idx4(k+1:k+n4(i)), 1:no_of_clusters);
    sd_train_features = [sd_train_features; h];
    k=k+n4(i);
end
%imds = imageDatastore(fullfile(img_path, categories), 'LabelSource', 'foldernames');

%Normalization in order to imp
for i=1:img_num
     poke_train_features(i,:) = poke_train_features(i,:)/sum(poke_train_features(i,:));
end
poke_train_features = cat(2,poke_train_features,color_feat);
for i=1:img_num
     cp_train_features(i,:) = cp_train_features(i,:)/sum(cp_train_features(i,:));
end

for i=1:img_num
     hp_train_features(i,:) = hp_train_features(i,:)/sum(hp_train_features(i,:));
end
for i=1:img_num
     sd_train_features(i,:) = sd_train_features(i,:)/sum(sd_train_features(i,:));
end

CPC = fitcknn(cp_train_features,CP_train,'NSMethod','exhaustive','Distance','euclidean');
HPC = fitcknn(hp_train_features,HP_train,'NSMethod','exhaustive','Distance','euclidean');
IDC = fitcknn(poke_train_features,ID_train,'NSMethod','exhaustive','Distance','euclidean');
SDC = fitcknn(sd_train_features,stardust_train,'NSMethod','exhaustive','Distance','euclidean');

CPC.NumNeighbors = 9;
HPC.NumNeighbors = 9;
IDC.NumNeighbors = 9;
SDC.NumNeighbors = 9;




save('model.mat','SDC','CPC','HPC','IDC','sdcodebook','hpcodebook','cpcodebook','pokecodebook','mapsaved');