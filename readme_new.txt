########

Things changed to run angles:
(1) added a function to load PrimaryAzimuth/PrimaryZenith in
./dataset/graph.py
./dataset/hd5.py
(2) changed last sigmoid in GraphConvolutionalNetwork to another ReLu in
./model/gcn.py
(3) changed loss function from binarycrossentropy to l1loss in
./train.py
(4) LR: 5e-3, was 5e-4


how to split to train,test and vali
(1) tun the transform_dataset.py under create_dataset dir.


