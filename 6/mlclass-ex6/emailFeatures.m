function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;

% Sort the word indices.
sorted_word_indices = sort(word_indices);

% Note that the below runs in only O(n) time as each index of x only considered
% once.
word_index_list_len = length(word_indices);
x = zeros(n, 1);
next_word_index = 1;
i = 1;
while 1	
	while (i ~= sorted_word_indices(next_word_index))
		i = i+1;
	end
	x(i) = 1;
	next_word_index = next_word_index + 1;
	if (next_word_index > word_index_list_len)
		% We've exhausted word indices list and are done searching.
		break;
	end
		
end
	 

end
