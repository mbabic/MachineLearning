function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

m = size(X, 1);	% m == number of training examples
n = size(X, 2); % n == number of features (not including 0th, constant feature)

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set options for call to fmincg
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Train each identifier in turn.	
for i = 1:num_labels
	[ret] = fmincg(@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
	    zeros(n + 1, 1), options);
	% Set ith row of all_theta to be the returned column vector 
	all_theta(i, :) = ret';	
end
end
