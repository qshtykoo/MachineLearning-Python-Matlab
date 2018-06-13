close all 
clear all


a=gendats(100); %generate 100 random samples
%multiply feature 2 by a factor of 10
% scaled_data = [a.DATA(:,1),10*a.DATA(:,2)];

figure;
gscatter(a.DATA(:,1),a.DATA(:,2),a.LABELS);
% figure;
% gscatter(scaled_data(:,1),scaled_data(:,2),a.LABELS);

% [f1,theta1,y1] = weak_learner(a.DATA, a.LABELS);
% % [f2,theta2,y2] = weak_learner(scaled_data, a.LABELS);
% test_result1 = hypo(f1,theta1,y1,a.DATA,a.LABELS);
% % test_result2 = hypo(f2,theta2,y2,scaled_data,a.LABELS);
% 
% error1 = 0;
% %error2 = 0;
% for i = 1:length(test_result1)
%     if test_result1(i) ~= a.LABELS(i)
%         error1 = error1 + 1;
%     end
% %     if test_result2(i) ~= a.LABELS(i)
% %         error2 = error2 + 1;
% %     end
% end
% 
% error_rate1 = error1 / length(a.LABELS);
% error_rate2 = error2 / length(a.LABELS);

iter_num = 100;

for T = 1:iter_num
    ada = adaboost_ver2(a.DATA, a.LABELS, T); %fit adaboost model with training dataset
    test_result_boost1 = classify(ada, a.DATA); %predict test dataset and output labels
    error_boost1 = sum(abs(test_result_boost1 - a.LABELS'));
    error_rate_boost1(T) = error_boost1/length(a.LABELS);   
end
