close all 
clear all


[MIL_dataset,bag_of_instances,baglabs] = gendatmilsival();

for i = 1:120
    %leave one out cross validation
    test_index = [i]; 
    train_index = exclude([1:120],test_index);
    
    %partitioning original dataset into test dataset and train dataset
    test_bag = {bag_of_instances{[test_index]}};
    test_baglab = baglabs([test_index]);
    train_bag = {bag_of_instances{[train_index]}};
    train_baglab = baglabs([train_index]);
    
    %generate dataset using bags2dataset
    MIL_train_data = bags2dataset(train_bag',train_baglab);
    MIL_test_data = bags2dataset(test_bag',test_baglab);
    W = fisherc(MIL_train_data);
    
    error(i) = classify_based_on_combineinstlabels(MIL_test_data, test_baglab, W);
end

average_error = mean(error);
    