function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
v = [0.01,0.03,0.1, 0.3, 1, 3, 10, 30];
error = zeros(8,8);

for i = 1:length(v)
    C = v(i);
    for j = 1:length(v)
        sigma = v(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        error(i,j) = mean(double(predictions ~= yval));

    end
end
[~,I] = min(error(:));
[I_row, I_col] = ind2sub(size(error),I);
C = v(I_row);
sigma = v(I_col);

end
