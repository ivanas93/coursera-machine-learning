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

hx = X * theta; % mxn * nx1 = mx1
errors_vector = hx - y; % mx1
sqerror = (errors_vector).^2;
theta(1) = 0;
reg = (lambda/(2*m)) * sum(theta.^2);
J = ((1/(2*m)) * sum(sqerror)) + reg;
grad = ((1/m) * (X' * (errors_vector))) + ((lambda/m) * theta);

% NB gradient is equal to the sum of X * errors_vector, scaled by 1/m and with
% the regularization addition of lambda/m * theta. Note that Theta(1) is 0
% so does not contribute.
% mxn' * mx1 = nx1 (same as Theta, which is what we want) + Theta scaled by 1/m
% but theta(1) is zero so does not change gradient of Theta(1) upon addition
% try adding "%" before the reg term and you'll see that the gradient of theta(1)
% does not change

grad = grad(:);

end