function test_result = hypo(f, theta, y, test_set, label)

TestSet = test_set(:,f);
test_result = zeros(length(TestSet),1);
label_name = unique(label);

if y == 0
      for i = 1:length(TestSet)
           if TestSet(i) <= theta
                test_result(i,1) = label_name(1);
           else
                test_result(i,1) = label_name(2);
           end
       end
else
       for i = 1:length(TestSet)
           if TestSet(i) > theta
                test_result(i,1) = label_name(1);
           else
                test_result(i,1) = label_name(2);
           end
       end
end


