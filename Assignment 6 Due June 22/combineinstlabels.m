function selected_label = combineinstlabels(labels)

    %labels should be a column vector 

    labelnames = unique(labels);
    out = [labelnames, histc(labels(:),labelnames)];
    [max_num,ind] = max(out(:,2));
    selected_label = labelnames(ind);


end