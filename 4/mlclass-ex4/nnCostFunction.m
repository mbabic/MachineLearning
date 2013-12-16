function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Add column of ones to X corresponding to bias unit.
X = [ones(m, 1) X];

% h is a matrix storing the results of the 
h = zeros(num_labels, m);

% _y is a (num_labels) sized column vector created from the result of each
% training example to be used in the calculation of the cost function.
_y = zeros(num_labels, 1);

% ones_vector is a convenience variable -- an column vector of 1's to be used
% in the calculation of the cost function (such that such a vector does not have
% to be created in each iteration of the main for loop)
ones_vector = ones(num_labels, 1);

% Accumulated gradient vars.
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));

% Done in a loop one example at a time to ease the implementation of the 
% backward propogation algorithm.
for i = 1:m
	a_1 = X(i, :)';

	z_2 = Theta1 * a_1;
	a_2 = sigmoid(z_2);
	a_2 = [1; a_2];		% add bias unit to a_2

	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);

	% We now create appropriate y vector for this training example.
	for j = 1:num_labels
		if (y(i, 1) == j)
			_y(j, 1) = 1;
		else
			_y(j, 1) = 0;
		end
	end

	% Calculate the next summand in the cost function.
	J = J + sum( ...
	    ((-_y) .* log(a_3)) - ... 
	    ((ones_vector - _y) .* log(ones_vector - a_3)) ...
	    );

	% Now we proceed with the back propogation calculations.
	
	% First, we calculate delta values for the output layer.
	delta_3 = a_3 - _y;
	% Then, we calculate delta values for the hidden layer.
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z_2]);
	delta_2 = delta_2(2:end);	% remove delta_2_0 term


	% Now we accumulate the gradients.
	Delta_2 = Delta_2 + (delta_3 * a_2');
	Delta_1 = Delta_1 + (delta_2 * a_1');

end
J = J / m;

% Now we regularize the cost function using the regularization parameter lambda.
Theta1_sum = sum(sum(Theta1(:, 2 : end) .^ 2));
Theta2_sum = sum(sum(Theta2(:, 2 : end) .^ 2));
J = J + ( (lambda / (2 * m)) * (Theta1_sum + Theta2_sum) );

% Accumulate the gradients.
Theta1_grad = Delta_1 ./ m;
Theta2_grad = Delta_2 ./ m;

% Regularize gradients.
Theta1_grad(:, 2:end) = Theta1_grad(:, 2: end) ...
	    + ((lambda / m) .* (Theta1(:, 2:end)));
Theta2_grad(:, 2 : end) = Theta2_grad(:, 2:end) ...
	    + ((lambda / m) .* (Theta2(:, 2:end)));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
