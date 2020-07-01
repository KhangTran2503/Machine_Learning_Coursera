function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
%X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));
%numcol = size(X,2);
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% size of mu and sigma is 1 x 2      
% mu(1,1) = mean(X(:,1)) 
% mu(1,2) = mean(X(:,2))
% sigma(1,1) = std(X(:,1))
% sigma(1,2) = std(X(:,2))
%mu    = mean(X_norm);
%sigma = std(X_norm);

% Standardization: x(next) = (x - mu)/sigma
%for iter = 1:numcol  
%	  X_norm(:,iter) = (X_norm(:,iter) - mu(iter)) / sigma(iter);
%end;
X_norm = X;
numColumns = size(X, 2);

mu = mean(X);
sigma = std(X);
for i = 1:numColumns
    X_norm(:,i) = (X(:, i) - mu(i)) / sigma(i);
end;


% ============================================================

end
