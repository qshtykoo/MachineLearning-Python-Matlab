function [instances,lab] = extractinstances(lab,img1,index)

%     if strcmp(type, 'b')
%         img1 = readImages(image_sequence_num,'b');
%     elseif strcmp(type, 'a')
%         img1 = readImages(image_sequence_num,'a');
%     end
%     
%     lab = im_meanshift(img1, 35);
%     image_sequence_num
    
    instances = unique(lab);
    out = [instances, histc(lab(:),instances)];
    [max_num,ind] = max(out(:,2));
    back_ground_value = instances(ind);
    
    num_of_segments = length(instances);
%     amin = min(min(lab));
%     amax = max(max(lab));
%     I = mat2gray(lab,[amin amax]);
%     figure;
%     imshow(I);

    [x,y]=size(lab);
    
    count_b = 1;
    segment = 1;
%     R_b = 0;
%     G_b = 0;
%     B_b = 0;
%     for i = 1:x
%          for j = 1:y
%                  if lab(i,j) == back_ground_value
%                         R_b(count_b) = img1(i,j,1);
%                         G_b(count_b) = img1(i,j,2);
%                         B_b(count_b) = img1(i,j,3);
%                         count_b = count_b + 1;   
%                   end
%           end
%     end
    
    while segment <= num_of_segments
    R_f = [];
    G_f = [];
    B_f = [];
    count_f = 1;
        for i = 1:x
            for j = 1:y
                     if lab(i,j) == segment
                           R_f(count_f) = img1(i,j,1);
                           G_f(count_f) = img1(i,j,2);
                           B_f(count_f) = img1(i,j,3);
                           count_f = count_f + 1;
                     end
            end
        end
    R{segment} = R_f;
    G{segment} = G_f;
    B{segment} = B_f;
    segment = segment + 1;
    end
    
%     mean_R_b = mean(R_b);
%     mean_G_b = mean(G_b);
%     mean_B_b = mean(B_b);
%     
%     R_f = R_f(any(R_f,2),:);
%     G_f = G_f(any(G_f,2),:);
%     B_f = B_f(any(B_f,2),:);
    
    for i = 1:num_of_segments
        if ismember(index,[1:6]) || ismember(index,[37:42])
            if i ~= back_ground_value
                R_o = R{i};
                G_o = G{i};
                B_o = B{i};
                instances_of_object(i,:) = [mean(R_o) mean(G_o) mean(B_o)];
            end
        else
            R_o = R{i};
            G_o = G{i};
            B_o = B{i};
            instances_of_object(i,:) = [mean(R_o) mean(G_o) mean(B_o)];
        end
    end
    
    if ismember(index,[1:6]) || ismember(index,[37:42])
        instances_of_object = instances_of_object(any(instances_of_object,2),:);
    end

    
    instances = instances_of_object;
end