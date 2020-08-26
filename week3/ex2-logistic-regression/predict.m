function p = predict(theta, X)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);


htheta = sigmoid(X * theta);

% Logical comparison converts it into 1's and 0's
p = htheta >= 0.5;



% =========================================================================


end
