clear all
close all

[MIL_dataset,bag_of_instances,baglabs, MIL_dataset_b, MIL_dataset_a] = gendatmilsival();
W = fisherc(MIL_dataset); %train Fisher classifier
% W = liknonc(MIL_dataset);
%[E,C] = testc(MIL_dataset*W); %test Fisher classifier


b_lab = baglabs(1:60);
a_lab = baglabs(61:120);

error_b_to_a = classify_based_on_combineinstlabels(MIL_dataset_b, b_lab, W);
error_a_to_b = classify_based_on_combineinstlabels(MIL_dataset_a, a_lab, W);

error = classify_based_on_combineinstlabels(MIL_dataset, baglabs, W);