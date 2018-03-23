clc; close all;

[trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
[valX, valY, valy] = LoadBatch('Dataset/data_batch_2.mat');
[testX, testY, testy] = LoadBatch('Dataset/test_batch.mat');

[d, N] = size(trainX);
[K, ~] = size(trainY);
m = 50; % number of hidde nodes

mean_X = mean(trainX,2);
trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
valX = valX - repmat(mean_X, [1, size(valX,2)]);
testX = testX - repmat(mean_X, [1, size(testX,2)]);

theta = InitParam(m, d, K);

[P, H] = EvaluateClassifier(trainX, theta);

% sub-functions
function [X, Y, y] = LoadBatch(filename)
dataSet = load(filename);
X = double(dataSet.data)'/255;
y = double(dataSet.labels+1)';
N = length(y);
K = max(y);
Y = zeros(K, N);
for i = 1:N
    Y(y(i), i) = 1;
end
end

function theta = InitParam(m, d, K)
b1 = zeros(m,1);
b2 = zeros(K,1);
W1 = randn(m,d) + 0.001;
W2 = randn(K,m) + 0.001;

theta = {W1, W2, b1, b2};
end

function [P, H] = EvaluateClassifier(X, theta)

s1 = bsxfun(@plus, theta{1}*X, theta{3});
H = max(0, s1);
s = bsxfun(@plus, theta{2}*H, theta{4});
P = softmax(s);
end

function [theta_grad] = ComputeGradients(X, Y, P, H, theta, lambda)
N = size(X,2);
theta_grad = {};
for k=1:2
    grad_W = zeros(size(theta{k}));
    grad_b = zeros(size(theta{k+2}));
for i=1:N
    g = (-Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-(P(:,i)*P(:,i)'));
    grad_b = grad_b + g';
    grad_W = grad_W + g'*X(:,i)';
end
grad_b = grad_b/N;
grad_W = grad_W/N + 2*lambda*theta{k};
theta_grad{k} = grad_W;
theta_grad{k+2} = grad_b;
end
end
