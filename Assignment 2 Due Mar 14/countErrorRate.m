function Error = countErrorRate(Mdl, featureDataNew, text_data)

% TestDataSet = zeros();
% TestDataSetClass = string();
% for N = [0, 10, 20, 40, 80, 160, 320, 640]
%    N2 = randi(19020-640);
%    for i = N2:N2+N
%        for j = 1:10
%     TestDataSet(i-N2+1,j) = featureDataNew(i,j);
%     TestDataSetClass(i-N2+1) = text_data(i);
%        end
%    end
%     c = randperm(length(featureDataNew),N);  
%     TestDataSet= M(c,:);  % output matrix

    
    test_data_class = string();
    for i = 1:length(text_data)
        test_data_class(i) = text_data(i);
    end
    result = predict(Mdl, featureDataNew); %apply LDA classifier
    %calculate error
    error = 0;
    for i = 1:length(result)
        if result(i)~= test_data_class(i)
            error = error + 1;
        end
    end
    Error = error/length(result);

