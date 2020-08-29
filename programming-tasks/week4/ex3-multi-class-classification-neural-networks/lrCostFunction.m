function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%           sigmoid(X * theta)
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
h = sigmoid(X * theta);
theta(1,1) = 0;
J = (1/m) * ((-y)' * log(h) - (1-y)' * log(1 -h)) + (lambda/(2*m)) * (theta' * theta);
grad = (1/m) * X' * (h - y) + (lambda/m) * theta;
grad(1,1) = (1/m) * X'(1,:) * (h - y)

grad = grad(:);

end
