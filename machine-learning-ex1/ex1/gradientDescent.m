function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    num_features = length(theta);
    theta_tmp = zeros(num_features, 1);
   
    for j=1:num_features
        
        sigma = 0;
        for i = 1:m
          sigma = sigma + (X(i,:) * theta - y(i)) * X(i, j);
        end
        
        theta_tmp(j) = theta(j) - (alpha / m) * sigma; 

    end

    theta = theta_tmp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
