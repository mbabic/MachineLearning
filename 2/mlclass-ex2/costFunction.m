function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

J = (1 / m) * sum ( (-y) .* (log(sigmoid(X*theta))) - ... 
	(ones(m, 1) - y) .* (log(ones(m, 1) - sigmoid(X*theta))) )


grad = (1 / m) * (X' * (sigmoid(X*theta) - y))

end
