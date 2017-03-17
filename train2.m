% You can change anything you want in this script.
% It is provided just for your convenience.
clear; clc; close all;



img_path = './train/';
%class_num = 30;
%img_per_class = 80;
img_num = 1086;
label_train = zeros(img_num,1);
concatenated_feat=[];
n = zeros(img_num,1);
%Defining the number of clusters for the K means
no_of_clusters=5;
imagefiles = dir('./train/*.jpg');
if isempty(imagefiles)
    imagefiles = dir('./train/*.png');
end

nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(currentfilename);
   images{i} = currentimage;
   label=currentfilename(1:3);
   label_train(i,1)=label;
   image = imread([img_path,folder_dir(i+2).name,'/',imagefiles(j).name]);
   imshow(image);
   image = rgb2gray(image);
   points = detectSURFFeatures(image);
   %points = detectHarrisFeatures(I)
   [feat, validpoints] = extractFeatures(image, points.selectStrongest(300),'SURFSize', 128);
   %[feat, validpoints] = extractFeatures(image, points.selectStrongest(300),'SURFSize', 128);
   concatenated_feat=[concatenated_feat;feat];
   n((i-1)*img_per_class+j) = size(feat,1);
        %cell{(i-1)*img_per_class+j, 1} = feat;
        %feat_train((i-1)*img_per_class+j,:) = feature_extraction(img);
end
    
%Clustering algorithm
[idx, codebook] = kmeans(concatenated_feat, no_of_clusters,'MaxIter',1000');

train_features = [];
k=0;
%Generating the histogram or vector of visual words for each image
for i=1:img_num
    h = hist(idx(k+1:k+n(i)), 1:no_of_clusters);
    train_features = [train_features; h];
    k=k+n(i);
end
%imds = imageDatastore(fullfile(img_path, categories), 'LabelSource', 'foldernames');

%Normalization in order to imp
for i=1:img_num
     train_features(i,:) = train_features(i,:)/sum(train_features(i,:));
end

save('model.mat','train_features','codebook','label_train');