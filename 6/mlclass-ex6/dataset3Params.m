function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C_vec = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

C_opt = C_vec(1);
sigma_opt = sigma_vec(1);
opt_err = realmax;
for i = 1:length(sigma_vec)
	sigma_test = sigma_vec(i);
	for j = 1:length(C_vec)
		C_test = C_vec(j);
		% Train SVM on test C and sigma values.
		model = svmTrain(X, y, C_test, @(x1, x2) ...
		    gaussianKernel(x1, x2, sigma_test));

		% Get predictions from trained SVM model on cross valuation
		% data set.
		predictions = svmPredict(model, Xval);

		% Get prediction error.
		err = mean(double(predictions ~= yval));

		if err < opt_err
			opt_err = err;
			sigma_opt = sigma_test;
			C_opt = C_test;
		end 
	end
end	

sigma = sigma_opt;
C = C_opt;

end
