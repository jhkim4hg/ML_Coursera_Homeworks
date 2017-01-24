function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Linear regression

h = X * theta;
J1 = (1 / (2 * m)) * sum((h-y) .^ 2);

theta0 = theta(2:end);
J2 = (lambda / (2 * m)) * sum(theta0 .^ 2);

J = J1 + J2;

% Gradient

grad_temp = (1 / m) * (X' * (h - y));

grad(1) = grad_temp(1);
grad(2:end) = grad_temp(2:end) + (lambda / m) * theta(2:end);

% =========================================================================

grad = grad(:);

end
