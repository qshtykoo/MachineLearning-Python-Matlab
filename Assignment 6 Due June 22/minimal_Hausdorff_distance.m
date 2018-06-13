function distance = minimal_Hausdorff_distance(bag1, bag2)


length2 = size(bag2,1);

% min_instance_B = min(cell2mat(bag2));

% for i = 1:length1 %iterate through bag1
%     instance_A = bag1(i,:);
%     for j = 1:length2 %iterate through bag2
%     instance_B = bag2(j,:);
%     distances(i,j) = norm(instance_A - instance_B);
%     end
% end
% 
for j = 1:length2 %iterate through bag2
    instance_B = bag2(j,:);
    distances(j,:) = vecnorm(bag1' - instance_B');
end


min_val = min(min(distances));
distance = min_val;
