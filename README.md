# Trash-Classification
A project of Artificial Intelligence class: using CNN networks to classify certain types of trash.


%HUONG DAN TRANSFER LEARNING PRETRAINED NETWORKS
B1: chay file LoadData.m de tai tap anh Training va Validate vao matlab
B2: chay file LoadResnet18.m hoac cac mang khac theo y muon
B3: chay file TrainNetwork.net de thuc hien training.
B4: chay file ValidateNetwork.m de tinh Do chinh xac top 1 va top 5
B5: save lai net da train bang lenh save('<ten muon save>', 'net')


%HUONG DAN VALIDATE HOAC TEST NETWORK DA TRAIN LAI
B1: load mang da train vao matlab va dat ten bang "net"
B2: load data bang file LoadDatasets.m
B2: chay file ValidateNetwork.m hoac TestNetwork.m de tinh Do chinh xac top 1 va top 5 cho tap anh Validation hoac Test

%HUONG DAN ENSEMBLE KET QUA
B1: load cac tap anh vao workspace su dung LoadDatasets.m
B2: load 3 mang muon ensemble vao workspace bang LoadEnsemble.m
B3: chay file ValidateEnsemble.m hoac TestEnsemble.m de tinh do chinh xac tren tap anh validation hoac test

