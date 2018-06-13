clear all
close all

a=gendats(100); %generate 100 random samples
weights = 0.01*ones(1,100);
original_weights = weights;
weights(1) = 10000*weights(1);
p = weights/sum(weights);
original_p = original_weights/sum(original_weights);

figure;
gscatter(a.DATA(:,1),a.DATA(:,2),a.LABELS);

[f1,theta1,y1] = weak_learner_weighted(a.DATA, a.LABELS, p');
[f2,theta2,y2] = weak_learner_weighted(a.DATA, a.LABELS, original_p');


