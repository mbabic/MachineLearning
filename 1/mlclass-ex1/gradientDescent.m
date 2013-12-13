function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
multiplier = alpha / m;
for iter = 1:num_iters
	temp_theta0 = theta(1) - multiplier * sum( X(:, 1)' * (X * theta - y));
	temp_theta1 = theta(2) - multiplier * sum( X(:, 2)' * (X * theta - y)); 
	theta = [temp_theta0; temp_theta1];
   	J_history(iter) = computeCost(X, y, theta);

end

end
