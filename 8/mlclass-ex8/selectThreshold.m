function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
	% Get binary vector such that i'th entry == 1 iff we predicted that the i'th
	% training example was an anamoly.	
	predictions = (pval < epsilon);
	
	% Calculate the number of true positive (i.e., the number of examples that are
	% correctly classified as anomalies given the current value of epsilon).
	true_positives = sum((predictions == 1) & (predictions == yval));
	% Calculate the number of false negatives (i.e., the number of examples that
	% were anomalies but classified as normal given the current value of epsilon).
	false_negatives = sum((predictions == 0) & (yval == 1));

	total_positives = sum(predictions == 1);

	% Protect against division by zero.
	if ( (total_positives == 0) || (true_positives + false_negatives == 0) )
		continue;
	end
	precision = true_positives / (sum(predictions == 1));
	recall = true_positives / (true_positives + false_negatives);

	F1 = (2 * precision * recall) / (precision + recall);

    	if F1 > bestF1
       		bestF1 = F1;
      		bestEpsilon = epsilon;
   	end
end

end
