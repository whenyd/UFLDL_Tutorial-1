function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);
m = size(data,2);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%hypothesis

h = theta * data;
h = bsxfun(@minus, h, max(h, [], 1));
h = exp(h);
h = bsxfun(@rdivide,h,sum(h));

%cost
temp = groundTruth .* log(h);
temp = sum(temp(:));
cost = -1 / m * temp + lambda / 2 * (theta(:)' * theta(:));

%grad
g = groundTruth - h;
for i = 1:numClasses
    thetagrad(i,:) = -1/m .* sum(data .* repmat(g(i,:),inputSize,1),2)' + lambda .* theta(i,:);
end

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

