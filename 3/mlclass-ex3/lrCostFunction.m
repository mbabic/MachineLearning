function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 	(1 / m) * sum ( (-y) .* (log(sigmoid(X*theta))) - ... 
	(ones(m, 1) - y) .* (log(ones(m, 1) - sigmoid(X*theta))) ) + ...
	(lambda / (2 * m) ) * (sum(theta .^ 2) - theta(1)^2);


% We store grad(1) into a temporary variable as theta_0 is not meant to be
% regularized but we still want to implement the equation in a vectorized
% way.
grad = (1 / m) * (X' * (sigmoid(X*theta) - y));
tmp = grad(1);

% Now we regularize the gradient
grad = grad + ((lambda / m) * theta);

% Un-regularize the 0th gradient.
grad(1) = tmp;

end
