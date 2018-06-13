function [MIL_dataset, bags, baglabs, MIL_dataset_b, MIL_dataset_a] = gendatmilsival()

load('lab1.mat');
load('lab2.mat');

for i = 1:60
    img1 = readImages(i,'b');
    instances_b = extractinstances(lab1{i},img1,i);
    num_instances_b(i) = size(instances_b,1);
    instances_b = double(instances_b);
    instance_matrix_b{i} = instances_b;
    
    bag_b_total{i} = [];
    for j = 1:num_instances_b(i)
        bag_b_total{i} = [bag_b_total{i}; instances_b(j,:)];
    end
    baglab_b(i) = 1;
    
    img2 = readImages(i,'a');
    instances_a = extractinstances(lab2{i},img2,i);
    num_instances_a(i) = size(instances_a,1);
    instances_a = double(instances_a);
    instance_matrix_a{i} = instances_a;
    
    bag_a_total{i} = [];
    for j = 1:num_instances_a(i)
        bag_a_total{i} = [bag_a_total{i}; instances_a(j,:)];
    end
    baglab_a(i) = 2;
    
    i
end

bags = [bag_b_total';bag_a_total'];
baglabs = [baglab_b';baglab_a'];
MIL_dataset = bags2dataset(bags,baglabs);
MIL_dataset_b = bags2dataset(bag_b_total',baglab_b');
MIL_dataset_a = bags2dataset(bag_a_total',baglab_a');

num_b = length(bag_b_total);
num_a = length(bag_a_total);

C1_b = MIL_dataset.DATA(1:num_b,:);
C1_a = MIL_dataset.DATA(num_b+1:num_b+num_a,:);

figure;
scatter3(C1_b(:,1),C1_b(:,2),C1_b(:,3),[],'r');
hold on
scatter3(C1_a(:,1),C1_a(:,2),C1_a(:,3),[],'b');
hold off

