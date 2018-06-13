%obtain 25 samples as
function Mdl = trainLDAClassifier(sizeOfData, featureDataNew, text_data)
N = sizeOfData ;
trainingDataSet = zeros();
trainingDataSetClass = char();

for i = 1:1:25
    N1 = randi(N);
    for j = 1:1:10
    trainingDataSet(i,j) = featureDataNew(N1,j);
    trainingDataSetClass(i) = text_data(N1);
    end
end

Mdl = fitcdiscr(trainingDataSet,trainingDataSetClass');

% Mdl = fitcdiscr(trainingDataSet,trainingDataSetClass, 'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus'));