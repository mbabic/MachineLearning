function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

centroids = zeros(K, n);

% TODO: the implementation below is stupid.  We should only have to walk
% the vector idx once to get all the information we need, but I don't have time
% at the moment to figure out how to create arrays of structures in octave in
% order to store the info as idx is walked.
 
for k = 1:K
	index_vec = zeros(0, 1);

	% Find indices of data points assigned to centroid k and stores indices
	% in index_vec.
	for i = 1:m
		if (idx(i) == k)
			index_vec = [index_vec; i];
		end
	end

	% We now proceed to calculate the mean of the data points associated
	% with centroid k.

	% We calculate the mean of each column (i.e., each feature) of X from
	% each row (i.e., training example) which is assigned to centroid k.
	means = mean(X(index_vec, :));
	centroids(k, :) = means;
end

end

