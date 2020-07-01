function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta)
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X*theta);
for i = 1:m 
  J = J + (-y(i)*log(h_theta(i)) - (1 - y(i))*log(1 - h_theta(i)))/m;
endfor

% regularize
for i = 2:n
   J = J + (lambda/(2*m))*(theta(i)^2);
endfor 

% grad
for i = 1:n
    h_y = h_theta - y;
    grad(i) = (1/m)*(X(:,i)'*h_y);
    if (i > 1)
       grad(i) = grad(i) + (lambda/m)*theta(i);
    endif
endfor
% =============================================================

end
