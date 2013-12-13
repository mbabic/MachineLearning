function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
m = size(X, 1); % Number of training examples

% Threshold = 0.5 (i.e., if theta' * (ith training example feature vecotr) >=
% 0.5 we predict a positive result)
p = ( sigmoid(X * theta) >= 0.5 )
end
