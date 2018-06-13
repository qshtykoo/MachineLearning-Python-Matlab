function Error = CrossValidation(sizeOfTrainingData, trainingDataSet, trainingDataSetClass)
%apply 5-fold cross validation on the 25 samples
for N2 = 1:5:sizeOfTrainingData-4
    testdsF = zeros();
    testdscF = string();
    trainingdsF = zeros();
    trainingdscF = string();

    %build testing dataset from the 25 samples
    for i = N2:1:N2+4
        for j = 1:1:10
            testdsF(i-N2+1,j) = trainingDataSet(i,j);
            testdscF(i-N2+1) = trainingDataSetClass(i);
        end
    end
    %build training dataset from the 25 samples
    for i = 1:N2-1
        for j = 1:10
        trainingdsF(i,j) = trainingDataSet(i,j);
        trainingdscF(i) = trainingDataSetClass(i);
        end
    end
    for i = N2+5:25
        for j = 1:10
        trainingdsF(i-5,j) = trainingDataSet(i,j);
        trainingdscF(i-5) = trainingDataSetClass(i);
        end
    end
    %train LDA classifier
    Mdl = fitcdiscr(trainingdsF,trainingdscF);
    %apply LDA classifier
    result = predict(Mdl,testdsF);
end 
    %calculate error
    error = 0;
    for i = 1:length(result)
        if result(i)~= testdscF(i)
            error = error + 1;
        end
    end
    Error = error/length(result);
