% Exercise: Aggregating Algorithm (AA)

clear all;
coin_data = load('coin_data.mat');
coin_data = struct2cell(coin_data);

s0 = coin_data{4,1};
s = coin_data{3,1};
r = coin_data{2,1};
symbols_str = coin_data{5,1};

d = 5;
n = 213;

for t=1:n
% compute adversary movez z_t
         z_t(t,:) = -log(r(t,:));
% compute strategy p_t (see slides)
     if t == 1
         % compute cumulative losses of experts --> L_t
         L_t(t,:) = z_t(t,:);
         p_t(t,:) = ones(1,5).*0.2;
     else
         L_t(t,1) = sum(z_t(1:t,1));
         L_t(t,2) = sum(z_t(1:t,2));
         L_t(t,3) = sum(z_t(1:t,3));
         L_t(t,4) = sum(z_t(1:t,4));
         L_t(t,5) = sum(z_t(1:t,5));
         C_tm1 = sum( exp(-L_t(t-1,:)) );
         p_t(t,1) = exp(-L_t(t-1,1))./C_tm1;
         p_t(t,2) = exp(-L_t(t-1,2))./C_tm1;
         p_t(t,3) = exp(-L_t(t-1,3))./C_tm1;
         p_t(t,4) = exp(-L_t(t-1,4))./C_tm1;
         p_t(t,5) = exp(-L_t(t-1,5))./C_tm1;
     end  
end
%total loss of the experts 
for i = 1:d
     loss_E(i) = L_t(213,i);
end
% compute loss of strategy p_t --> l_m
     [l_m, g] = mix_loss(p_t', r');  

% compute regret --> Rn
 min_z_t = sum(z_t(:,1));
 min_d = 1;
 
 for j = 1:d
     if min_z_t >= sum(z_t(:,j))
         min_z_t = sum(z_t(:,j));
         min_d = j;
     end
 end
 
 Rn = sum(l_m) - min_z_t;
 
% compute total gain of investing with strategy p_t --> w_t/w_1
for t = 1:n
    w_tOverw_t(t,1) = sum(p_t(t,:).*r(t,:));
end
totalGain = prod(w_tOverw_t);
%% plot of the strategy p and the coin data

% if you store the strategy in the matrix p (size n * d)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(p_t)
legend(symbols_str)
title('rebalancing strategy AA')
xlabel('date')
ylabel('confidence p_t in the experts')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
