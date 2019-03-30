function [train_predict, test_predict] = q2a()
data = load('mnist_m.mat');
xtrain = data.train_data;
length = size(xtrain, 2);
ytrain = data.train_classlabel;
ytrain(ytrain == 1 | ytrain == 4) = 1;
ytrain(ytrain ~= 1) = 0;
ytrain1 = cast(ytrain, 'double');
ytrain2 = ~ytrain1;
xtest = data.test_data;
ytest = data.test_classlabel;
ytest(ytest == 1 | ytest == 4) = 1;
ytest(ytest ~= 1) = 0;
ytest1 = cast(ytest, 'double');
radius = zeros(length, length);
for i=1:length
    for j=1:length
        radius(i, j) = norm(xtrain(:, i) - xtrain(:, j));
    end
end
phi = exp(-radius.^2/2);
% for first output neuron
w1 = phi \ ytrain1';
% for second output neuron
w2 = phi \ ytrain2';
test_length = size(ytest, 2);
test_radius = zeros(test_length, length);
for i=1:test_length
    for j=1:length
        test_radius(i, j) = norm(xtest(:, i) - xtrain(:, j));
    end
end
train_predict1 = exp(-radius.^2/2) * w1;
train_predict2 = exp(-radius.^2/2) * w2;
train_predict = (train_predict1 - train_predict2) > 0;
test_predict1 = exp(-test_radius.^2/2) * w1;
test_predict2 = exp(-test_radius.^2/2) * w2;
test_predict = (test_predict1 - test_predict2) > 0;

train_error = sum(abs(ytrain1 - train_predict')) / length;
test_error = sum(abs(ytest1 - test_predict')) / test_length;
display(train_error);
display(test_error);

TrAcc = zeros(1, 1000);
TeAcc = zeros(1, 1000);
thr = zeros(1, 1000);
TrN = length;
TeN = test_length;
for i=1:1000
    t = (max(train_predict) - min(train_predict)) * (i-1)/1000 + min(train_predict);
    thr(i) = t;
    TrAcc(i) = (sum(ytrain1(train_predict<t)==0) + sum(ytrain1(train_predict>=t)==1)) / TrN;
    TeAcc(i) = (sum(ytest1(test_predict<t)==0) + sum(ytest1(test_predict>=t)==1)) / TeN;
end
plot(thr, TrAcc, '.-', thr, TeAcc, '^-');
legend('tr', 'te');
end