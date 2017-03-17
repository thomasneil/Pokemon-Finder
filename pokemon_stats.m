function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats (img, model)
% Please DO NOT change the interface
% INPUT: image; model(a struct that contains your classification model, detector, template, etc.)
% OUTPUT: ID(pokemon id, 1-201); level(the position(x,y) of the white dot in the semi circle); cir_center(the position(x,y) of the center of the semi circle)
%
%save('model.mat','sd_train_features','cp_train_features','hp_train_features','poke_train_features','sdcodebook','hpcodebook','cpcodebook','pokecodebook','CP_train','HP_train','stardust_train','ID_train');

% Replace these with your code
model=load('model.mat');

map=model.mapsaved;
poke_codebook=model.pokecodebook;
hp_codebook=model.hpcodebook;
cp_codebook=model.cpcodebook;
sd_codebook=model.sdcodebook;
[x1,y1,z1] = size(img);
poke_level=imcrop(img,[0,0,y1,(x1/2)]);
poke_level=im2bw(poke_level,0.95);
%imshow(poke_level);
disk_size=uint8(0.0055*y1);
dilate_size=strel('disk',5);
eroded=imerode(poke_level,dilate_size);
dilated=imdilate(eroded,dilate_size);
%imshow(eroded);
range1=uint8(y1*0.001);
range2=uint8(y1*0.011);
try
[centers,radii,metric]=imfindcircles(eroded,[range1 range2],'ObjectPolarity','bright','Sensitivity',0.92);

catch exception
    [centers,radii,metric]=imfindcircles(eroded,[3 5],'ObjectPolarity','bright','Sensitivity',0.92);
    
end    
imshow(eroded);
viscircles(centers,radii,'EdgeColor','b');

bigcircle=im2bw(img,0.95);
range1=uint8(y1*0.41);
range2=uint8(y1*0.30);
try
    [bigcenters,bigradii,bigmetric]=imfindcircles(bigcircle,[range1 range2],'ObjectPolarity','bright','Sensitivity',0.92);
    [max_big, big_index]=max(bigradii);
    cir_center = bigcenters(big_index,:);
    if isempty(cir_center)
    cir_center=[(x1*0.35) (y1*0.5)];
    end
catch exception
    cir_center=[(x1*0.35) (y1*0.5)];
end 


%cir_center;
  try
        img_gray = rgb2gray(img);
        
    catch exception
        %[X, map] = imread(img);
        %image(X);
        img = ind2rgb(img,map);
    end
img=imresize(img,[1000,600]);

%     [centers,radii] = imfindcircles(img,[249,251]);
%     [centers2,radii2] = imfindcircles(img,[1,2]);
poke_color=imcrop(img,[1,83,596,330]);
   
cpimg=imcrop(img,[1,83,596,330]);
    
bwcp=im2bw(cpimg,0.95);
    
    
    
    
    img_gray = rgb2gray(img);
        
%     catch exception
%         [X, map] = imread([img_path,img_dir(i).name]);
%         %image(X);
%         rgb = ind2rgb(X,map);
%         
%         img_gray=rgb2gray(rgb);
%         
%     end
    
    R = poke_color(:,:,1);
    G = poke_color(:,:,2);
    B = poke_color(:,:,3);
    tR=graythresh(R)/255;
    tG=graythresh(G)/255;
    tB=graythresh(B)/255;
    R = mean(R(:))/255;
    G = mean(G(:))/255;
    B = mean(B(:))/255;
    
%     color_feat(i,:) = [R G B];
poke=imcrop(img_gray,[1,83,596,330]);
    cpimg=imcrop(img_gray,[1,1,598,100]);
    hpimg=imcrop(img_gray,[1,505,598,50]);
    sdimg=imcrop(img_gray,[300,780,120,60]);
% imshow(poke);
% imshow(cpimg);
% imshow(hpimg);
% imshow(sdimg);
pokepoints = detectSURFFeatures(poke);
pointshp = detectSURFFeatures(hpimg);
%pointscp = detectSURFFeatures(cpimg);
pointscp = detectSURFFeatures(bwcp);
pointssd = detectSURFFeatures(sdimg);
[pokemonfeat, idpoints] = extractFeatures(poke, pokepoints.selectStrongest(300),'SURFSize', 128);
    [cpfeat, cppoints] = extractFeatures(bwcp, pointscp.selectStrongest(100),'SURFSize', 128);
    [hpfeat, hppoints] = extractFeatures(hpimg, pointshp.selectStrongest(100),'SURFSize', 128);
    [sdfeat, sdpoints] = extractFeatures(sdimg, pointssd.selectStrongest(100),'SURFSize', 128);
%[feature,validPoints] = extractFeatures(img, points.selectStrongest(300),'SURFSize',128);


t=zeros(1,70);
for k=1:size(pokemonfeat,1)
    
    for r=1:size(poke_codebook,1)
        a(r) = sqrt(sum((pokemonfeat(k,:) - poke_codebook(r,:)) .^ 2));
        
    end
[M,I] = min(a);
t(I) = t(I)+1;
end
poke_feat = t/sum(t);

poke_feat=[poke_feat R G B tR tG tB];
%poke_feat=[poke_feat R G B];        
t=zeros(1,70);
for k=1:size(cpfeat,1)
    
    for r=1:size(cp_codebook,1)
        a(r) = sqrt(sum((cpfeat(k,:) - cp_codebook(r,:)) .^ 2));
        
    end
[M,I] = min(a);
t(I) = t(I)+1;
end
cp_feat = t/sum(t);

t=zeros(1,70);
for k=1:size(hpfeat,1)
    
    for r=1:size(hp_codebook,1)
        a(r) = sqrt(sum((hpfeat(k,:) - hp_codebook(r,:)) .^ 2));
        
    end
[M,I] = min(a);
t(I) = t(I)+1;
end
hp_feat = t/sum(t);

t=zeros(1,70);
for k=1:size(sdfeat,1)
    
    for r=1:size(sd_codebook,1)
        a(r) = sqrt(sum((sdfeat(k,:) - sd_codebook(r,:)) .^ 2));
        
    end
[M,I] = min(a);
t(I) = t(I)+1;
end
sd_feat = t/sum(t);


ID_Classifier=model.IDC;
CP_Classifier=model.CPC;
HP_Classifier=model.HPC;
SD_Classifier=model.SDC;


CP=predict(CP_Classifier, cp_feat);
HP=predict(HP_Classifier, hp_feat);
ID=predict(ID_Classifier, poke_feat);
stardust=predict(SD_Classifier, sd_feat);

% ID = 1;
% CP = 123;
% HP = 26;
% stardust = 600;
[max_level, max_index]=max(radii);
max_center = centers(max_index,:);
%cir_center = centers;
cir_center
level=max_center;
if isempty(level)
    level=[100,500];
end   
end
