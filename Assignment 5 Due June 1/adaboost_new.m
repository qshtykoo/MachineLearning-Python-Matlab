classdef adaboost_new
     properties
     f(1,:)
     theta(1,:)
     y(1,:)
     beta_vec(1,:)
     iter_num
     weights(1,:)
     label_name(1,:)
     end
         methods
             function obj = adaboost_new(TrainSet, train_label, T)
                 if nargin == 3
                    [num_labeled_example, num_feature] = size(TrainSet);
                    count = 1;
                    %Intialize the weight vector
                    weight_vector = (1/num_labeled_example)*ones(num_labeled_example,1);
                    while count <= T
                        %normalize the weight vector
                        p = weight_vector/sum(weight_vector);
                        %provide the weak_learner with distribution p
                        [f,theta, y] = weak_learner_weighted(TrainSet, train_label, p);
                        %store the outputs of weak_learner
                        f_vec(count) = f;
                        theta_vec(count) = theta;
                        y_vec(count) = y;   
                        % Calculating error
                        test_result = hypo(f, theta, y, TrainSet, train_label);
                        error = sum(p.*abs(test_result - train_label));
                        beta(count) = error / (1 - error);    
                        %update the weight vector
                        for i = 1:length(weight_vector)
                            weight_vector(i) = weight_vector(i)*(beta(count)^(1-abs(test_result(i)-train_label(i))));
                        end
                        count = count + 1;
                    end
                  obj.f = f_vec;
                  obj.theta = theta_vec;
                  obj.y = y_vec;
                  obj.beta_vec = beta;
                  obj.iter_num = T;
                  obj.weights = weight_vector;
                  obj.label_name = unique(train_label);
                  end
             end
             function test_result_boost = classify(obj, test_data)
             %output the hypothesis
                    for j = 1:max(size(test_data))
                        sum_left = 0;
                        sum_right = 0;
                        for i = 1:obj.iter_num
                            test_result = hypo(obj.f(i), obj.theta(i), obj.y(i), test_data(j,:), obj.label_name);
                            sum_left = sum_left + log(1/obj.beta_vec(i))*test_result;
                            sum_right = sum_right + 0.5*log(1/obj.beta_vec(i));
                        end
                        if sum_left >= sum_right
                            test_result_boost(1,j) = 1;
                        else
                            test_result_boost(1,j) = 0;
                        end
                    end
             end
         end
end
