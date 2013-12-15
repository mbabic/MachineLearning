function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

%       are in rows, then, you can use max(A, [], 2) to obtain the max 

% Note that the below calculation can be done on one line but has been broken
% into several steps (for clarity/pedagogical reasons)

% First we multiply X and the transpose of all_theta to create an
% m by num_classifiers sized matrix such that entry (i, j) corresponds to
% multiplication of the features of the ith training example by the weigths
% of theta for the jth classifier.  Therefore, the index of the maximum value
% in the ith row corresponds to our prediction of the classification of the
% ith training examples (e.g., if row 10 has max value at index 8, then we
% predict that the 10th training example is the digit '8', recall that 
% classifier 10 corresponds to the digit 0) 

tmp = (X * all_theta');

% Now we take the index of the max element from each row (we store the values
% in garbage variable _ and the indices in p) 
[_, p] = max(tmp, [], 2);

% Clear garbage variable '_' from memory.
clear('_');

end
