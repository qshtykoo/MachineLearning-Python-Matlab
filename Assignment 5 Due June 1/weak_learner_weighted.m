function [f,theta, y] = weak_learner_weighted(trainSet, label, weights)

[x,fea_num] = size(trainSet);
label_name = unique(label);

for i = 1:fea_num
    f = i;
    count = 1;
    max_theta = max(trainSet(:,i));
    min_theta = min(trainSet(:,i));
    
    num_pace = 30; %define the number of possible threshold values to iterate through
    pace = (max_theta - min_theta) / num_pace; %define the pace the iterate through
        distance = []; %initialize the distance as zero
        param = [];
        theta = min_theta;
        %iterate through all available thetas in feature i
        while theta <= max_theta
            result1 = zeros(x,1);
            result2 = zeros(x,1);
            for j = 1:x   
                    if trainSet(j,i) <= theta
                        result1(j,1) = label_name(1); %label w1
                        result2(j,1) = label_name(2); %label w2
                    else
                        result1(j,1) = label_name(2);
                        result2(j,1) = label_name(1);
                    end
            end
            if sum(abs(result1' - label').*weights') >= sum(abs(result2' - label').*weights')
                y = 1; %'>'
                result = sum(abs(result2' - label').*weights');
            else
                y = 0; %'<'
                result = sum(abs(result1' - label').*weights');
            end
            distance(1,count) = result;
            param(:,count) = [f; theta; y]; 
            count = count + 1;
            if pace == 0 
                break;
            else
                theta = theta + pace;
            end
        end
        [min_val, ind] = min(distance);
        min_distance(i) = min_val; %storing the minimum error of feature i
        param_op(:,i) = param(:,ind); %storing the corresponding parameters f, theta and y
end

[min_error, index] = min(min_distance);
f = param_op(1,index);
theta = param_op(2,index);
y = param_op(3,index);