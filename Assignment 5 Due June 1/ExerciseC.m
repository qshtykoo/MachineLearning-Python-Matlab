clear all
close all

%First generate some artificial 2D data, and split it in a training and test set. 
%In this example we use 20% for training, and the remaining 80% will be used for testing.
A = gendats(100);

scaled_data = [10*A.DATA(:,1), A.DATA(:,2)];

[f1,theta1,y1] = weak_learner(A.DATA, A.LABELS);

[f2,theta2,y2] = weak_learner(scaled_data, A.LABELS);