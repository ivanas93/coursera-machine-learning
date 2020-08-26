function J = computeCost(X, y, theta)
m = length(y); % number of training examples

J = 0;

h = X * theta;
squaredErrors = (h - y) .^ 2;
J = (1 / (2 * m)) * sum(squaredErrors);

end
