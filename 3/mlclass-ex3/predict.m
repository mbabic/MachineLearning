function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add column of 1's to X (bias unit)
X = [ones(m, 1) X];

z_2 = (Theta1 * X');
a_2 = sigmoid(z_2);
a_2 = a_2';		% take transpose to make a_2 an (m) by (number of 
			% nodes in hidden layer) sized matrix such that we
			% can apply the same steps in the calculation
			% of the next layer as we did in the calculation of
			% this one

% Add column of 1's to a_2 (bias unit)
a_2 = [ones(m, 1) a_2];

% Calculate final layer stuff.
z_3 = Theta2 * a_2';
a_3 = sigmoid(z_3);

a_3 = a_3';
[_, p] = max(a_3, [], 2);
clear('_');


% =========================================================================


end
