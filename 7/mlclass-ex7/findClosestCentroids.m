function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X, 1);

idx = zeros(size(X,1), 1);


for i = 1:m
	min_dist = realmax;
	closest_centroid = -1;
	for k = 1:K
		dist = sum((X(i, :) - centroids(k, :)) .^ 2);
		if (dist < min_dist)
			min_dist = dist;
			closest_centroid = k;
		end
	end
	idx(i) = closest_centroid;
end

end

