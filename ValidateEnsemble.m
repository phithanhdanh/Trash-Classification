%% Classify with each network

inputSize = net1.Layers(1).InputSize;
augimdsValidation = augmentedImageDatastore(inputSize(1:2), ...
    imdsValidation,'ColorPreprocessing','gray2rgb');
[YPred1,scores1] = classify(net1,augimdsValidation);

inputSize = net2.Layers(1).InputSize;
augimdsValidation = augmentedImageDatastore(inputSize(1:2), ...
    imdsValidation,'ColorPreprocessing','gray2rgb');
[YPred2,scores2] = classify(net2,augimdsValidation);

inputSize = net3.Layers(1).InputSize;
augimdsValidation = augmentedImageDatastore(inputSize(1:2), ...
    imdsValidation,'ColorPreprocessing','gray2rgb');
[YPred3,scores3] = classify(net3,augimdsValidation);

%% Ensemble results
scores = (scores3+scores2+scores1)/3;
labels = net1.Layers(end).Classes;
[M,idx] = max(scores,[],2);
clear YPred;
YPred(:,1) = labels(idx,1);

%% Validate ensembled results

YValidation = imdsValidation.Labels; 

%Top 1 accuracy
Top1Accuracy = mean(YPred == YValidation)
%Top 5 accuracy
[n,m] = size(scores);  
idx = zeros(m,n); 
for i=1:n  
    [~,idx(:,i)] = sort(scores(i,:),'descend');  
end  
idx = idx(1:5,:);  
top5Classes = net.Layers(end).ClassNames(idx);  
top5count = 0;  
for i = 1:n  
    top5count = top5count + sum(YValidation(i,1) == top5Classes(:,i));  
end  
top5Accuracy = top5count/n 

%Display some top 1 predicted images
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

