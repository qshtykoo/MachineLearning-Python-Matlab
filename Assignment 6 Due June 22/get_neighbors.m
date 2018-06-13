function [neighbors, neighbor_labels] = get_neighbors(train_set, train_label, test_set, k)


len_train = length(train_set);
len_test = length(test_set);

for i = 1:len_test
    bag_test = test_set{i};
    distance = [];
    for j = 1:len_train
        bag_train = train_set{j};
        distance(1,j) = minimal_Hausdorff_distance(bag_train, bag_test);
    end
    [k_min_values, k_ind] = mink(distance,k); %find the smallest k elements
    neighbors{i} = {train_set{k_ind}};
    neighbor_labels{i} = train_label(k_ind);
end








