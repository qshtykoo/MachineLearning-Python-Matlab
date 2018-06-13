close all
clear all


% [MIL_dataset,bags,baglabs] = gendatmilsival();
load('MIL_dataset');
load('bags');
load('baglabs');

x_n = MIL_dataset.DATA;

for sigma = 1:100
    for i = 1:length(bags) %bag level
        bag_i = bags{i};
        for k = 1:length(x_n) %instance level
            max_value = calculate_s(bag_i(1,:), x_n(k,:), sigma);
            for j = 2:size(bag_i,1) %instance inside bag level
                if max_value <= calculate_s(bag_i(j,:), x_n(k,:), sigma)
                    max_value = calculate_s(bag_i(j,:), x_n(k,:), sigma);
                end
            end
            s(k,1) = max_value;
        end
        m_B_i(:,i) = s;
    end

    DATA_MILES = prdataset(m_B_i',baglabs); %produce dataset
    
    average_error(sigma) = prcrossval(DATA_MILES,liknonc); %leave-one-out cross validation
    sigma
end

