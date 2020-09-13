function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

%LEARNINGCURVE Generates the train and cross validation set errors needed
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular,
%       it returns two vectors of the same length - error_train and
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.

% Number of training examples
m = size(X, 1);

% return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

theta = [1; 1];
for i = 1:m
  X_test = X(1:i, :);
  y_test = y(1:i);
  thetas = trainLinearReg(X_test, y_test, lambda);


  error_train(i) = linearRegCostFunction(X_test, y_test, thetas, 0); % get cost and gradient

  error_val(i) = linearRegCostFunction(Xval, yval, thetas, 0);

end

end