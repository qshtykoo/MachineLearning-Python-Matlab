clear all
close all

a = gendats(100);

% labels = str2num(a.LABELS);
labels = a.LABELS;

p = .7 ;     % proportion of rows to select for training
N = size(a.DATA,1);  % total number of rows 
tf = false(N,1) ;   % create logical index vector
tf(1:round(p*N)) = true;    
tf = tf(randperm(N));   % randomise order
dataTraining = a.DATA(tf,:);
labelTraining = labels(tf);
dataTesting = a.DATA(~tf,:);
labelTesting = labels(~tf);

figure;
gscatter(a.DATA(:,1),a.DATA(:,2),a.LABELS);


[f1,theta1,y1] = weak_learner(dataTraining, labelTraining);
test_result1 = hypo(f1,theta1,y1,dataTesting,labelTesting);

error1 = 0;
for i = 1:length(test_result1)
    if test_result1(i) ~= labelTesting(i)
        error1 = error1 + 1;
    end
end
error_rate1 = error1 / length(labelTesting);



iter_num = 100;

for T = 1:iter_num
    ada = adaboost_ver2(dataTraining, labelTraining, T); %fit adaboost model with training dataset
    test_result_boost = classify(ada, dataTesting); %predict test dataset and output labels
    error_boost = sum(abs(test_result_boost - labelTesting'));
    error_rate_boost(T) = error_boost/length(labelTesting); 
end

figure;
plot([1:T], error_rate_boost);
xlabel('iteration number') % x-axis label
ylabel('error rate') % y-axis label
legend('Test Error')

figure;
plot([1:length(labelTraining)],ada.weights)

