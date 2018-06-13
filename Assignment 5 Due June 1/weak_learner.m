function [f,theta, y] = weak_learner(trainSet, label)

[x,fea_num] = size(trainSet);
label_name = unique(label);


for i = 1:fea_num
    f = i;
    count = 1;
    max_theta = max(trainSet(:,i));
    min_theta = min(trainSet(:,i));
    distance = []; %initialize the distance as zero
    param = [];
    %iterate through all available thetas in feature i
    for theta = min_theta:1:max_theta
        result1 = zeros(x,1);
        result2 = zeros(x,1);
        for j = 1:x   %x is the number of samples
            if trainSet(j,i) <= theta    
                %result1 and result2 are used to distinguish different results using different signs '<' and '>' 
                %different signs are represented by the value of y. When y
                %is 0, the sign is '<', when y is 1, the sign is '>'
                result1(j,1) = label_name(1); %label w1 
                result2(j,1) = label_name(2); %label w2 
            else
                result1(j,1) = label_name(2);
                result2(j,1) = label_name(1);
            end
        end
        if pdist2(result1', label') >= pdist2(result2', label')
            y = 1; %'>'
            result = pdist2(result2',label');
        else
            y = 0; %'<'
            result = pdist2(result1',label');
        end
        distance(1,count) = result;
        param(:,count) = [f; theta; y]; 
        count = count + 1;
    end
    [min_val, ind] = min(distance);
    min_distance(i) = min_val; %storing the minimum error of feature i
    param_op(:,i) = param(:,ind); %storing the corresponding parameters f, theta and y
end

[min_error, index] = min(min_distance);
f = param_op(1,index);
theta = param_op(2,index);
y = param_op(3,index);