function [J, grad] = costFunction(theta, X, y)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

htheta = sigmoid(X * theta);
J = 1/m * sum(-y .* log(htheta) - (1 - y) .* log(1 - htheta));
grad = (X'*(htheta - y))/m;

% =============================================================

end
