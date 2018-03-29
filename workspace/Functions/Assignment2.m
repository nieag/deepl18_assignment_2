clear all;
rng(400);
[trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
[valX, valY, valy] = LoadBatch('Dataset/data_batch_2.mat');
[testX, testY, testy] = LoadBatch('Dataset/test_batch.mat');

[d, N] = size(trainX);
[K, ~] = size(trainY);
m = 50; % number of hidden nodes

mean_X = mean(trainX,2);
trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
valX = valX - repmat(mean_X, [1, size(valX,2)]);
testX = testX - repmat(mean_X, [1, size(testX,2)]);


[b, W] = InitParam(m, d, K);
% [b_grad, W_grad] = ComputeGradients(trainX(1:300,1), trainY(:,1), W, b, lambda);
% [b_gradn, W_gradn] = ComputeGradsNumSlow(trainX(1:300,1), trainY(:,1), W, b, lambda, 1e-5);
%
% w1_grad_diff = max(max(abs(W_grad{1}-W_gradn{1})))
% w2_grad_diff = max(max(abs(W_grad{2}-W_gradn{2})))
% b1_grad_diff = max(max(abs(b_grad{1}-b_gradn{1})))
% b2_grad_diff = max(max(abs(b_grad{2}-b_gradn{2})))

% [Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(trainX(:, 1:100), trainY(:, 1:100), valX(:, 1:100), valY(:, 1:100), GDparams, W, b, lambda);

% Search for hyper params

lambda = 0.000001;
GDparams.n_batch=100;
GDparams.eta=0.01; % Learning rate
GDparams.rho=0.9; %momentum
GDparams.decay=0.95; % Learning rate decay
GDparams.n_epochs = 5;

[Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);

% e_min = log10(0.0220);
% e_max = log10(0.0235);
% l_min = log10(9.731e-4); 
% l_max = log10(9.739e-4); 

% Coarse search range
% e_range = {log10(0.01), log10(0.03)};
% l_range = {log10(10e-7), log10(10e-1)};

% Fine search range
e_range = {log10(0.025), log10(0.026)};
l_range = {log10(3.613e-05), log10(3.617e-05)};
% %
n_runs = 50;
disp("Starting run")
params = HyperParamSearch(e_range, l_range, trainX, trainY, valX, valY, valy, GDparams, n_runs);

save('storeMatrix.mat','params');

% Optimal hyper param
% eta_opt = 0.397244491243286;
% lambda_opt = 2.038499529402853e-08;
% 
% GDparams.n_epochs = 10;
% GDparams.eta = eta_opt;
% 
% [Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda_opt);
% test_acc = ComputeAccuracy(testX, testy, Wstar, bstar)

% sub-functions
function params = HyperParamSearch(e_range, l_range, trainX, trainY, valX, valY, valy, GDparams, n_runs)
for i=1:n_runs
    disp("Starting search run:"+ num2str(i));
    [d, N] = size(trainX);
    [K, ~] = size(trainY);
    m = 50; % number of hidden nodes
    
    [b, W] = InitParam(m, d, K);
    
    e= e_range{1} + (e_range{2} - e_range{1})*rand(1, 1);
    eta = 10^e;
    GDparams.eta = eta;
    
    l = l_range{1} + (l_range{2} - l_range{1})*rand(1, 1);
    lambda = 10^l;
    
    [Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);
    
    acc = ComputeAccuracy(valX, valy, Wstar, bstar);
    disp("Accuracy: " + num2str(acc));
    params(i, 1) = eta;
    params(i, 2) = lambda;
    params(i, 3) = acc;
end
end

function acc = ComputeAccuracy(X, y, W, b)
P = EvaluateClassifier(X, W, b);
[~, kStar] = max(P);
correct = kStar==y;
acc = sum(correct)/length(correct);
end

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

function [b, W] = InitParam(m, d, K)
b1 = zeros(m,1);
b2 = zeros(K,1);
W1 = 0.001*randn(m,d);
W2 = 0.001*randn(K,m);

W = {W1, W2};
b = {b1, b2};
end

function J = ComputeCost(X, Y, W, b, lambda)
P = EvaluateClassifier(X, W, b);
D = size(X, 2);
Wij = sum(sum(W{1}.^2,1),2) + sum(sum(W{2}.^2,1),2);
lcross = -log(sum(Y.*P));
J = (1/D)*sum(lcross)+lambda*Wij;
end

function [P, H, s1] = EvaluateClassifier(X, W, b)
s1 = bsxfun(@plus, W{1}*X, b{1});
H = max(0, s1);
s2 = bsxfun(@plus, W{2}*H, b{2});
P = softmax(s2);
end

function [b_grad, W_grad] = ComputeGradients(X, Y, W, b, lambda)
N = size(X,2);
[P, H, s1] = EvaluateClassifier(X, W, b);
dldb2 = zeros(size(b{2})); dldb1 = zeros(size(b{1}));
dldw2 = zeros(size(W{2})); dldw1 = zeros(size(W{1}));
for i=1:N
    g = (-Y(:,i)'/(Y(:,i)'*P(:,i)))*(diag(P(:,i))-(P(:,i)*P(:,i)'));
    
    % Second layer
    dldb2 = dldb2 + g';
    dldw2 = dldw2 + g'*H(:,i)';
    
    g = g*W{2}; %propagate backwards
    g = g*diag(s1(:,i)>0); %propagate backwards
    
    % First layer
    dldb1 = dldb1 + g';
    dldw1 = dldw1 + g'*X(:,i)';
end

b_grad = {dldb1/N, dldb2/N};
W_grad = {dldw1/N + 2*lambda*W{1}, dldw2/N + 2*lambda*W{2}};
end

function [Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda)
n_batch = GDparams.n_batch;
eta = GDparams.eta;
n_epochs = GDparams.n_epochs;
rho = GDparams.rho;
decay = GDparams.decay;

N = size(trainX,2);
tL_saved=[];
vL_saved=[];

W_mom = {zeros(size(W{1})), zeros(size(W{2}))};
b_mom = {zeros(size(b{1})), zeros(size(b{2}))};

for i=1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        
        [b_grad, W_grad] = ComputeGradients(Xbatch, Ybatch, W, b, lambda);
        
        for k=1:2
            W_mom{k} = rho*W_mom{k} + eta*W_grad{k};
            b_mom{k} = rho*b_mom{k} + eta*b_grad{k};
            W{k} = W{k} - W_mom{k};
            b{k} = b{k} - b_mom{k};
            %             W{k} = W{k} - (eta*W_grad{k});
            %             b{k} = b{k} - (eta*b_grad{k});
        end
    end
    eta = decay*eta;
    trainLoss = ComputeCost(trainX, trainY, W, b, lambda);
    disp("Current training loss: " + num2str(trainLoss));
    tL_saved = [tL_saved;trainLoss];
    valLoss = ComputeCost(valX, valY, W, b, lambda);
    vL_saved = [vL_saved; valLoss];
end

Wstar = W;
bstar = b;
end

% numeric gradient slow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

%quick
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

c= ComputeCost(X, Y, W, b, lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end
