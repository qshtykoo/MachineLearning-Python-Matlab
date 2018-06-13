function [citer_labels, neighbor_ind] = get_citers(total_set, total_label, test_index, k)

len_train = length(total_set);

for i = 1:len_train
    bag_train_1 = total_set{i};
    distance = [];
    for j = 1:len_train
        bag_train_2 = total_set{j};
        distance(1,j) = minimal_Hausdorff_distance(bag_train_2, bag_train_1);
    end
    [k_min_values, k_ind] = mink(distance,k); %find the smallest 3 elements
    neighbor_ind(i,:) = k_ind;
%     neighbor_labels(i,:) = total_label(k_ind);
end

count_citer = 0;
for j = 1:len_train
      if ismember(test_index,neighbor_ind(j,:))
            count_citer = count_citer + 1;
            citers(count_citer,1) = j;
            citer_labels(count_citer,1) = total_label(j);
      end
end

if count_citer == 0
    citer_labels = 0;
end

end

