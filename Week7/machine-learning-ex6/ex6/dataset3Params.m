function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_list = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_list = C_list;

n = length(C_list);
m = length(sigma_list);

choice = zeros(n*m,3);
cnt = 0;

for i = 1:n
    for j = 1:m
        C_test = C_list(i);
        sigma_test = sigma_list(j);
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        predict_val = svmPredict(model,Xval);
        error_val = mean(double(predict_val ~= yval));
        cnt += 1;
        choice(cnt,:) = [error_val C_test sigma_test];
    end;
end;

[~,idxmin] = min(choice(:,1));
C = choice(idxmin,2);
sigma = choice(idxmin,3);





% =========================================================================

end
