function max_value = calculate_s(bag_i, x_n, sigma)
  
        component = (norm(bag_i-x_n))^2/(sigma^2);
        max_value = exp(-component);

end