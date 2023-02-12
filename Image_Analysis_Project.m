clc 
clear
if ~exist('MerchData1','dir')
    unzip('MerchData1.zip')
end
%% Feature extraction
image_datastore = imageDatastore('MerchData1','IncludeSubfolders',true,'LabelSource','foldernames');

net=resnet18;

[imdsTrain,imdsTest] = splitEachLabel(image_datastore,0.7,'randomized');


numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net)
ITrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
I = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'pool5';
VTrain = activations(net,ITrain,layer,'OutputAs','rows');
VTest = activations(net,I,layer,'OutputAs','rows');

whos featuresTrain
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
d=Euclidian_distance(VTest(1,:),VTest(2,:));
s= Similarity_Metric(d);
function distance=Euclidian_distance(x,y)
    x=x.';
    y=y.';
    distance = zeros(length(x) , length(y));
    for i = 1:length(x)
        for j = 1:length(y)
            if i ~= j
                distance(i,j) = sqrt((x(i)-x(j))^2 + (y(i)-y(j))^2);  
            end
        end
    end 
end
