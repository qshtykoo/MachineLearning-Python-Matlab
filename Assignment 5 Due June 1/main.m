clear all
close all

VecX = dlmread('optdigitsubset.txt');

N1 = 554;
N2 = 571;
sample_num = 50;
VecX1 = zeros(sample_num, 64);
VecX2 = zeros(sample_num, 64);

%% choose first 50 samples from each class serving as training dataset and the rest as test dataset
Vec1 = zeros(1,N1);
Vec2 = ones(1,N2);


for i = 1:1:sample_num
    for j = 1:1:64
        VecX1(i,j)=VecX(i,j);
    end
end

for i = 1:1:sample_num
    for j = 1:1:64
         VecX2(i,j)=VecX(i+N1,j);
    end
end

label = [Vec1(1:50), Vec2(1:50)];
label = label';
TrainSet = [VecX1;VecX2];

Testlabel = [Vec1(51:N1), Vec2(51:N2)];
Testlabel = Testlabel';
TestVec1 = VecX(51:N1,:);
TestVec2 = VecX(N1+51:N1+N2,:);
TestSet = [TestVec1;TestVec2];

[f,theta, y] = weak_learner(TrainSet, label);
test_result = hypo(f,theta,y,TestSet,label);

 %% Calculating error
error = 0;
for i = 1:length(Testlabel)
    if test_result(i) ~= Testlabel(i)
        error = error + 1;
    end
end

error_rate = error/length(Testlabel);

%% randomly choose 50 samples from each class serving as training dataset and the rest as test dataset
for T = 1:20 %iteration
    randindex_1  = randperm(N1,sample_num);
    randindex_2  = N1 + randperm(N2,sample_num);
    randindex = [randindex_1,randindex_2];
    label_total = [Vec1, Vec2];
     %create training dataset
    for i = 1:length(randindex)
        TrainSet_new(i,:) = VecX(randindex(i),:);
        TrainLabel(i) = label_total(randindex(i));
    end
     %create test dataset
     count = 1;
     for i = 1:length(label_total)
         if ismember(i,randindex) == 0
             TestSet_new(count,:) = VecX(i,:);
             TestLabel(count) = label_total(i);
             count = count + 1;
         end
     end
     
    [f,theta, y] = weak_learner(TrainSet_new, TrainLabel');
    test_result_new = hypo(f,theta,y,TestSet_new, TrainLabel);

    %% Calculating error
    error_new = 0;
    for i = 1:length(TestLabel)
        if test_result_new(i) ~= TestLabel(i)
            error_new = error_new + 1;
        end
    end

    error_rate_new = error_new/length(TestLabel);
    error_rate_vec(T) = error_rate_new;
end

average_error_rate_new = mean(error_rate_vec);
stadDev_error_rate_new = sqrt(var(error_rate_vec)); %standard deviation
figure;
plot([1:T],error_rate_vec);
hold on
plot([1:T],error_rate*ones(1,T));




    