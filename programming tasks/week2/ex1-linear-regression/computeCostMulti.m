function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples

J = 0;

h = X * theta; % Hypothesis function (Theta' * x = theta_0 + theta_1 * x_1)
squaredErrors = (h - y) .^ 2;
J = (1 / (2 * m)) * sum(squaredErrors);

end
