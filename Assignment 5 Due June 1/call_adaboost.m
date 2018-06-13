close all
clear all

VecX = dlmread('optdigitsubset.txt');

N1 = 554;
N2 = 571;
sample_num = 50;
VecX1 = zeros(sample_num, 64);
VecX2 = zeros(sample_num, 64);
Vec1 = zeros(1,N1); %label 0
Vec2 = ones(1,N2); %label 1

for i = 1:1:sample_num
    for j = 1:1:64
        VecX1(i,j)=VecX(i,j);
        VecX2(i,j)=VecX(i+N1,j);
    end
end

%training data labeled
label = [Vec1(1:50), Vec2(1:50)];
label = label';
TrainSet = [VecX1;VecX2];
%test dataset
test_data_1 = VecX(sample_num+1:N1, :);
test_data_2 = VecX(N1+sample_num+1:N1+N2,:);
test_label = [Vec1(sample_num+1:N1),Vec2(sample_num+1:N2)];
test_data = [test_data_1;test_data_2];



iter_num = 17;

for T = 1:iter_num
    ada = adaboost_new(TrainSet, label, T); %fit adaboost model with training dataset
    test_result_boost1 = classify(ada, test_data); %predict test dataset and output labels
    error_boost1 = sum(abs(test_result_boost1 - test_label));
    error_rate_boost1(T) = error_boost1/length(test_label);
    
    test_result_boost2 = classify(ada, TrainSet);
    error_boost2 = sum(abs(test_result_boost2 - label'));
    error_rate_boost2(T) = error_boost2/length(label);    
end

figure;
plot([1:T], error_rate_boost1);
hold on
plot([1:T], error_rate_boost2);
xlabel('iteration number') % x-axis label
ylabel('error rate') % y-axis label
legend('Test Error','Train Error')

figure;
plot([1:100],ada.weights);

figure;
subplot(2,2,1)
imshow(reshape(TrainSet(66,:),[8,8]));
subplot(2,2,2)
imshow(reshape(TrainSet(86,:),[8,8]));
subplot(2,2,3)
imshow(reshape(TrainSet(29,:),[8,8]));
subplot(2,2,4)
imshow(reshape(TrainSet(17,:),[8,8]));


