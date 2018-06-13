% Exercise: Online Gradient Descent (OGD)
close all;
clear all;
load coin_data;

a_init = [0.2, 0.2, 0.2, 0.2, 0.2]'; % initial action

n = 213; % is the number of days
d = 5; % number of coins

% we provide you with values R and G.
alpha = sqrt(max(sum(r.^2,2))); 
epsilon = min(min(r)); 
G = alpha/epsilon; 
R = 1; 

% set eta:
eta = R/(G*sqrt(n));

a = a_init; % initialize action. a is always a column vector

L = nan(n,1); % keep track of all incurred losses
A = nan(d,n); % keep track of all our previous actions

for t = 1:n
    
    % we play action a
    [l,g] = mix_loss(a,r(t,:)'); % incur loss l, compute gradient g
    
    A(:,t) = a; % store played action
    L(t) = l; % store incurred loss
    
    % update our action, make sure it is a column vector
    a = a - eta*g;
    
    % after the update, the action might not be anymore in the action
    % set A (for example, we may not have sum(a) = 1 anymore). 
    % therefore we should always project action back to the action set:
    a = project_to_simplex(a')'; % project back (a = \Pi_A(w) from lecture)

end

% compute total loss
totalLoss = sum(L);

% compute total gain in wealth
totalGain = exp(-totalLoss);
%Alternative method to calcultae totalGain
% Atry = A'
% for t = 1:n
%     w_tOverw_t(t,1) = sum(Atry(t,:).*r(t,:));
% end
% totalGain = prod(w_tOverw_t);

% compute best fixed strategy (you may make use of loss_fixed_action.m and optimization toolbox if needed)
cvx_begin
    variable a_fixed(1,5);
    Vec = [a_fixed; zeros(212,5)];
    minimize( norm(sum(Vec*r')) ); %the mix loss is defined as lm = -ln(a), therefore the component a should be minimized to get the optimal value of lm
    subject to
        sum(a_fixed) == 1;
        a_fixed(:) >= 0;
cvx_end


% compute regret 
[loss_fixed,g] = loss_fixed_action(a_fixed);
Rn = totalLoss - loss_fixed;

%% plot of the strategy A and the coin data

% if you store the strategy in the matrix A (size d * n)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(A')
legend(symbols_str)
title('rebalancing strategy OGD')
xlabel('date')
ylabel('investment action a_t')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
