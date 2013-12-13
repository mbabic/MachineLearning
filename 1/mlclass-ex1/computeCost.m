function J = computeCost(X, y, theta)
%COMPUTECOST COmput cost for linear regression.
%	J = COMPUTCOST(X, y, theta) computes the cost of using theta as the 
%	parameter for linear regression ot fit the data points in X and y

m = length(y); % number of training examples

J = (1 / (2 * m)) * sum(((X*theta) - y) .^ 2);

end
