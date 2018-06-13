clear all 
close all

% [MIL_dataset,bag_of_instances,baglabs] = gendatmilsival();
load('bag_of_instances');
load('baglabs');

%apply "leave one out" cross validation
c = 3;
k = 50;
for i = 1:120
        test_index = [i]; 
        train_index = exclude([1:120],test_index);

        %partitioning original dataset into test dataset and train dataset
        test_bag = {bag_of_instances{[test_index]}};
        test_baglab = baglabs([test_index]);
        train_bag = {bag_of_instances{[train_index]}};
        train_baglab = baglabs([train_index]);

        cKNN_classifier = citation_KNN(bag_of_instances, baglabs, train_bag, train_baglab, test_bag, test_index, k, c);
        test_label = output(cKNN_classifier);

        errors = sum(abs(test_baglab' - test_label));
        error_rate(i) = errors / length(test_label);
end
average_error = mean(error_rate);



