function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% vector H_theta = X*theta // H_theta(i) = X(i:)*theta
% h(x(i)) = sigmoid(theta' * x(i))

h_theta = sigmoid(X*theta);
for i = 1:m 
  J = J + (-y(i)*log(h_theta(i)) - (1 - y(i))*log(1 - h_theta(i)))/m;
endfor
 
% vector grad n x 1
for i = 1:n
  h_y = h_theta - y; % vector m x 1
  grad(i) = (1/m)*(X(:,i)'*h_y);
endfor 






% =============================================================

end
