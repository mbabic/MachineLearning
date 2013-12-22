function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user

% Calculate cost.
J = (1 / 2) * sum(sum((((X * Theta') - Y) .^ 2).* R)) + ...
    ( (lambda / 2) * sum(sum(Theta .^ 2)) ) + ...
    ( (lambda / 2) * sum(sum(X .^ 2)) );

% Calculate gradients.
common_matrix = ((X * Theta') - Y) .* R;

X_grad = (common_matrix * Theta) + (lambda .* X); 

Theta_grad = (common_matrix' * X) + (lambda .* Theta);

grad = [X_grad(:); Theta_grad(:)];

end