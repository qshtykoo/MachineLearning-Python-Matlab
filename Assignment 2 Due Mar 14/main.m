clear 
close all


sizeOfData = 19020;
[num_data, text_data] = xlsread('magic04data.csv');

sDeviation = zeros();
ave = zeros();
featureData = num_data;
for i = 1:1:10 %normalize data
    sDeviation(i) = std(featureData(:,i));
    ave(i) = sum(featureData(:,i))/sizeOfData;
end

featureDataNew = zeros();
for i = 1:1:10
    for j = 1:1:sizeOfData
        featureDataNew(j,i)=(featureData(j,i)-ave(i))/sDeviation(i);
    end
end

text_data = char(text_data);
% sDeviationT = zeros();
% for i = 1:1:10 %to test
%     sDeviationT(i) = std(featureDataNew(:,i));
% end
Error = zeros();
for i = 1:100
    Mdl = trainLDAClassifier(sizeOfData, featureDataNew, text_data);
    %cross validate the LDA classifier
    Error(i) = countErrorRate(Mdl, featureDataNew, text_data);
end
AverageError = mean(Error);
