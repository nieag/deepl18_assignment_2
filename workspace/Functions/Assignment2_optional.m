clear all;
rng(400);
%%% One batch for training
% [trainX, trainY, trainy] = LoadBatch('Dataset/data_batch_1.mat');
% [valX, valY, valy] = LoadBatch('Dataset/data_batch_2.mat');
% [d, N] = size(trainX);
% [K, ~] = size(trainY);
% mean_X = mean(trainX,2);
% trainX = trainX - repmat(mean_X, [1, size(trainX,2)]);
% valX = valX - repmat(mean_X, [1, size(valX,2)]);
%%% #################### %%%

%%% Multiple batches for final test
% [tx1, tY1, ty1] = LoadBatch('Dataset/data_batch_1.mat');
% [tx2, tY2, ty2] = LoadBatch('Dataset/data_batch_2.mat');
% [tx3, tY3, ty3] = LoadBatch('Dataset/data_batch_3.mat');
% [tx4, tY4, ty4] = LoadBatch('Dataset/data_batch_4.mat');
% [tx5, tY5, ty5] = LoadBatch('Dataset/data_batch_5.mat');
% [X_test, Y_test, y_test] = LoadBatch('Dataset/test_batch.mat');
% 
% 
% X_train = [tx1, tx2, tx3, tx4, tx5(:, 1:9000)];
% Y_train = [tY1, tY2, tY3, tY4, tY5(:, 1:9000)];
% y_train = [ty1, ty2, ty3, ty4, ty5(:, 1:9000)];
% 
% mean_X_train = mean(X_train, 2);
% X_train = X_train - repmat(mean_X_train, [1, size(X_train,2)]);
% 
% X_valid = tx5(:,9001:10000);
% X_valid = X_valid - repmat(mean_X_train, [1, size(X_valid,2)]);
% Y_valid = tY5(:,9001:10000);
% y_valid = ty5(:,9001:10000);
% 
% X_test = X_test - repmat(mean_X_train, [1, size(X_test,2)]);
%%% #################### %%%

%%% Gradient Comparison %%%
% CompareGradients(trainX, trainY, GDparams)
%%% #################### %%%

% Search for hyper params
% GDparams.n_batch=100;
% GDparams.rho=0.90; %momentum
% GDparams.decay=0.1; % Learning rate decay
% GDparams.n_epochs = 7;
% GDparams.activation = "ReLu";

% lambda=0;
% [b_grad, W_grad] = ComputeGradients(trainX, trainY, W, b, lambda, GDparams);
% CompareGradients(trainX, trainY, GDparams)
% tic
% [Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda, trainLossBound);
% toc
% Coarse search range
% e_range = {log10(0.008), log10(0.035)};
% l_range = {log10(0.000001), log10(0.1)};

% Fine search range
% e_range = {log10(0.0160), log10(0.025)};
% l_range = {log10(4.0e-04), log10(4.5e-04)};
% 
% n_runs = 50;
% disp("Starting run")
% params = HyperParamSearch(e_range, l_range, trainX, trainY, valX, valY, valy, GDparams, n_runs);
% [I, M] = max(params(:,3))
% save('storeMatrix.mat','params');

%%% Optimal hyper param
% [d, N] = size(X_train);
% [K, ~] = size(Y_train);
% m = 60; % number of hidden nodes
% 
% % From parameter search
% eta_opt = 0.023624927961652;
% lambda_opt = 3.247136269597346e-05;
% eta_opt = 0.020499374426818;
% lambda_opt = 4.013002959464533e-04;
% % 
% GDparams.n_batch=100;
% GDparams.rho=0.90; %momentum
% GDparams.decay=0.1; % Learning rate decay
% GDparams.n_epochs = 30;
% GDparams.eta = eta_opt;
% GDparams.activation = "ReLu";
% % 
% % 
% % [b, W] = InitParam(m, d, K);
% [b, W] = HeInitParam(m, d, K);
% [Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(X_train, Y_train, X_valid, Y_valid, GDparams, W, b, lambda_opt);
% test_acc = ComputeAccuracy(X_test, y_test, Wstar, bstar, GDparams)
% % 
% figure;
% plot(tL_saved); hold on;
% plot(vL_saved);
% title("Cross Entropy Loss for Training and Valdidation Data");
% xlabel("Epochs");
% ylabel("Cross entropy loss");
% legend("Training loss", "Validation loss");
% fnameMontage = sprintf('train_val_loss_ordinary_eta_%f_lambda_%f.png', eta_opt, lambda_opt);
% saveas(gcf, fnameMontage, 'png');
%%% #################### %%%

%%% Leaky ReLu testing
% CompareGradients(trainX, trainY, GDparams)

% GDparams.n_batch=100;
% GDparams.rho=0.90; %momentum
% GDparams.decay=0.95; % Learning rate decay
% GDparams.n_epochs = 10;
% GDparams.activation = "LeakReLu";

% Coarse search range
% e_range = {log10(0.008), log10(0.035)};
% l_range = {log10(0.000001), log10(0.1)};

% Fine search range
% e_range = {log10(0.027), log10(0.03)};
% l_range = {log10(1.56e-06), log10(3e-04)};
% 
% n_runs = 50;
% disp("Starting run")
% params = HyperParamSearch(e_range, l_range, trainX, trainY, valX, valY, valy, GDparams, n_runs);
% [I, M] = max(params(:,3))
% save('storeMatrix.mat','params');

% [d, N] = size(X_train);
% [K, ~] = size(Y_train);
% m = 50; % number of hidden nodes
% % 
% eta_opt = 0.029461459721245;
% lambda_opt = 5.773673941697393e-05;
% % % 
% GDparams.n_batch=100;
% GDparams.rho=0.90; %momentum
% GDparams.decay=0.95; % Learning rate decay
% GDparams.n_epochs = 30;
% GDparams.eta = eta_opt;
% GDparams.activation = "LeakReLu";
% % 
% [b, W] = InitParam(m, d, K);
% [Wstar, bstar, tL_saved, vL_saved] = MiniBatchGD(X_train, Y_train, X_valid, Y_valid, GDparams, W, b, lambda_opt);
% test_acc = ComputeAccuracy(X_test, y_test, Wstar, bstar, GDparams)
% % 
% figure;
% plot(tL_saved); hold on;
% plot(vL_saved);
% title("Cross Entropy Loss for Training and Valdidation Data");
% xlabel("Epochs");
% ylabel("Cross entropy loss");
% legend("Training loss", "Validation loss");
% fnameMontage = sprintf('train_val_loss_leak_eta_%f_lambda_%f.png', eta_opt, lambda_opt);
% saveas(gcf, fnameMontage, 'png');
%%% #################### %%%

%%% sub-functions
function params = HyperParamSearch(e_range, l_range, trainX, trainY, valX, valY, valy, GDparams, n_runs)
for i=1:n_runs
    disp("Starting search run:"+ num2str(i));
    [d, N] = size(trainX);
    [K, ~] = size(trainY);
    m = 60; % number of hidden nodes
    
    [b, W] = HeInitParam(m, d, K);
    
    e= e_range{1} + (e_range{2} - e_range{1})*rand(1, 1);
    eta = 10^e;
    GDparams.eta = eta;
    
    l = l_range{1} + (l_range{2} - l_range{1})*rand(1, 1);
    lambda = 10^l;
    
    [Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b, lambda);
    
    acc = ComputeAccuracy(valX, valy, Wstar, bstar, GDparams);
    disp("Accuracy: " + num2str(acc));
    params(i, 1) = eta;
    params(i, 2) = lambda;
    params(i, 3) = acc;
end
end

function acc = ComputeAccuracy(X, y, W, b, GDparams)
P = EvaluateClassifier(X, W, b, GDparams);
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

function CompareGradients(trainX, trainY, GDparams)
[d, N] = size(trainX(1:300, 1));
[K, ~] = size(trainY(:,1));
m = 50; % number of hidden nodes
lambda=0;

[b, W] = InitParam(m, d, K);
[b_grad, W_grad] = ComputeGradients(trainX(1:300,1), trainY(:,1), W, b, lambda, GDparams);
[b_gradn, W_gradn] = ComputeGradsNumSlow(trainX(1:300,1), trainY(:,1), W, b, lambda, 1e-5, GDparams);
[b_gradn_quick, W_gradn_quick] = ComputeGradsNum(trainX(1:300, 1), trainY(:,1), W, b, lambda, 1e-5, GDparams);    

w1_grad_diff_slow = max(max(abs(W_grad{1}-W_gradn{1})))
w2_grad_diff_slow = max(max(abs(W_grad{2}-W_gradn{2})))
b1_grad_diff_slow = max(max(abs(b_grad{1}-b_gradn{1})))
b2_grad_diff_slow = max(max(abs(b_grad{2}-b_gradn{2})))

w1_grad_diff_quick = max(max(abs(W_grad{1}-W_gradn_quick{1})))
w2_grad_diff_quick = max(max(abs(W_grad{2}-W_gradn_quick{2})))
b1_grad_diff_quick = max(max(abs(b_grad{1}-b_gradn_quick{1})))
b2_grad_diff_quick = max(max(abs(b_grad{2}-b_gradn_quick{2})))
end

function [b, W] = InitParam(m, d, K)
b1 = zeros(m,1);
b2 = zeros(K,1);
W1 = 0.001*randn(m,d);
W2 = 0.001*randn(K,m);

W = {W1, W2};
b = {b1, b2};
end

function [b, W] = HeInitParam(m, d, K)
b1 = zeros(m,1);
b2 = zeros(K,1);
stdevW1 = sqrt(2/d); % He-init of stdv for input layer
stdevW2 = sqrt(2/m);% He-init of stdv for hidden layer
W1 = stdevW1*randn(m,d);
disp(stdevW1);
disp(stdevW2);
W2 = stdevW2*randn(K,m);

W = {W1, W2};
b = {b1, b2};
end

function J = ComputeCost(X, Y, W, b, lambda, GDparams)
P = EvaluateClassifier(X, W, b, GDparams);
D = size(X, 2);
Wij = sum(sum(W{1}.^2,1),2) + sum(sum(W{2}.^2,1),2);
lcross = -log(sum(Y.*P));
J = (1/D)*sum(lcross)+lambda*Wij;
end

function [P, H, s1] = EvaluateClassifier(X, W, b, GDparams)
s1 = bsxfun(@plus, W{1}*X, b{1});
if GDparams.activation=="ReLu"
    H = max(0, s1); % ReLU activation
elseif GDparams.activation=="LeakReLu"
    H = max(0.1*s1, s1);
end
s2 = bsxfun(@plus, W{2}*H, b{2});
P = softmax(s2);
end

function [b_grad, W_grad] = ComputeGradients(X, Y, W, b, lambda, GDparams)
N = size(X,2);
[P, H, s1] = EvaluateClassifier(X, W, b, GDparams);
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

valLossOld = 10000;
N = size(trainX,2);
tL_saved=[];
vL_saved=[];
W_mom = {zeros(size(W{1})), zeros(size(W{2}))};
b_mom = {zeros(size(b{1})), zeros(size(b{2}))};
disp("Original training loss: " + num2str(ComputeCost(trainX, trainY, W, b, lambda, GDparams)));
for i=1:n_epochs
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        Xbatch = Xbatch + randn(size(Xbatch))*0.01 ; % Add random noise to trainng vector
        [b_grad, W_grad] = ComputeGradients(Xbatch, Ybatch, W, b, lambda, GDparams);
        
        for k=1:2
            W_mom{k} = rho*W_mom{k} + eta*W_grad{k};
            b_mom{k} = rho*b_mom{k} + eta*b_grad{k};
            W{k} = W{k} - W_mom{k};
            b{k} = b{k} - b_mom{k};
        end
    end
    if mod(i, 10)==0
        eta = eta*decay;
        disp("Eta: " + num2str(eta));
    end
    trainLoss = ComputeCost(trainX, trainY, W, b, lambda, GDparams);
    disp("Epoch: " + num2str(i) + " Current training loss: " + num2str(trainLoss));
    tL_saved = [tL_saved;trainLoss];
    valLoss = ComputeCost(valX, valY, W, b, lambda, GDparams);
    vL_saved = [vL_saved; valLoss];
    if valLoss<=valLossOld % Save the model that minimises the validation loss to avoid overfitting to training data
       Wstar = W;
       bstar = b;
    end
    valLossOld = valLoss;
end
disp("Final training loss: " + num2str(trainLoss));
end

% numeric gradient slow
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h, GDparams)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda, GDparams);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda, GDparams);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda, GDparams);
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda, GDparams);
        
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end

%quick
function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h, GDparams)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

c= ComputeCost(X, Y, W, b, lambda, GDparams);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda, GDparams);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda, GDparams);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end
end
