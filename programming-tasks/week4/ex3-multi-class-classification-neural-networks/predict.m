function p = predict(Theta1, Theta2, X)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
a2 = sigmoid( X * Theta1');
m = size(a2, 1);
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
[probability, p] = max(a3, [], 2);


end
