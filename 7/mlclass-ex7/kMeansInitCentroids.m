function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X

% Create a random permutation of the integers {1, ..., m} where m is the
% number of training examples.
randidx = randperm(size(X, 1));

% Sets the initial positions of the centroids to be K random training
% examples.
centroids = X(randidx(1:K), :);

end

