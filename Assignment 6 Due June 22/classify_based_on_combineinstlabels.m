function error = classify_based_on_combineinstlabels(MIL_dataset, baglabs, W)

LABELS = labeld(MIL_dataset,W);
bagid = getident(MIL_dataset,'milbag');
count_bag = 1;
[~, ind] = unique(bagid, 'rows');
bagid_unqiue = unique(bagid);
out = [bagid_unqiue, histc(bagid(:),bagid_unqiue)];
for i = 1:length(ind)
       begin_index = ind(i);
       num_of_same = out(i,2);
       labels_of_bag = [];
       count = 1;
       for j = begin_index:begin_index+num_of_same-1
           labels_of_bag(1,count) = LABELS(j);
           count = count + 1;
       end
       bags{count_bag} = labels_of_bag;
       count_bag = count_bag+1;
end

bags = bags';

predicted_labels = [];
for i = 1:length(bags)
    bag = bags{i};
    selected_label = combineinstlabels(bag');
%     predicted_labels = [predicted_labels; selected_label.*ones(1,length(bag))'];
    predicted_labels(i) = selected_label;
end

error = sum(abs(baglabs' - predicted_labels)) / length(predicted_labels);
%error = sum(abs(baglabs' - predicted_labels));