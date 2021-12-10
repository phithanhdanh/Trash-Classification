%% 
%LOAD PRETRAINED NETWORK
net = googlenet;
%analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;

%% 
%REPLACE FINAL LAYERS
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    fullyConnectedLayer(numClasses,"WeightLearnRateFactor",10,"BiasLearnRateFactor",10,"Name",'fc')
    softmaxLayer("Name",'softmax')
    classificationLayer("Name",'classoutput')];

lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,{'output','prob','loss3-classifier'}); 
lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');
%analyzeNetwork(lgraph)