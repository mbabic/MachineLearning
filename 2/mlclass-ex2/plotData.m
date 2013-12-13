function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.

% Find indices of positive and negative examples in the training set
positive_results = find(y == 1);
negative_results = find(y == 0);

% Plot the examples
% since positive_results is a vector of indices, the result of doing 
% X(positive_results, 1) is to select the lx1 subvector of X where each row
% corresponds to a row of the vector y in which the value 1 was stored and the
% column is the first column of the matrix X

plot(X(positive_results, 1), X(positive_results, 2), 'k+', 'LineWidth', 2, ... 
	'MarkerSize', 7);


plot(X(negative_results, 1), X(negative_results, 2), 'ko', 'MarkerFaceColor',...
	'y' , 'MarkerSize', 7);

hold off;

end
