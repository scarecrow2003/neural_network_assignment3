function [train_predict, test_predict] = q2ar()
data = load('mnist_m.mat');
xtrain = data.train_data;
train_length = size(xtrain, 2);
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
radius = zeros(train_length, train_length);
for i=1:train_length
    for j=1:train_length
        radius(i, j) = norm(xtrain(:, i) - xtrain(:, j));
    end
end
phi = [ones(train_length, 1) exp(-radius.^2/2)];
test_length = size(ytest, 2);
test_radius = zeros(test_length, train_length);
lambda = [0:0.01:0.9 1:1:10];
length = size(lambda, 2);
train_errors = zeros(1, length);
test_errors = zeros(1, length);
for i=1:length
    display("lambda: " + lambda(i));
    % for first output neuron
    w1 = (phi' * phi + lambda(i) * eye(size(phi, 2))) \ phi' * ytrain1';
    % for second output neuron
    w2 = (phi' * phi + lambda(i) * eye(size(phi, 2))) \ phi' * ytrain2';
    for j=1:test_length
        for k=1:train_length
            test_radius(j, k) = norm(xtest(:, j) - xtrain(:, k));
        end
    end
    train_predict1 = exp(-radius.^2/2) * w1(2:size(w1, 1)) + w1(1);
    train_predict2 = exp(-radius.^2/2) * w2(2:size(w2, 1)) + w2(1);
    train_predict = (train_predict1 - train_predict2) > 0;
    test_predict1 = exp(-test_radius.^2/2) * w1(2:size(w1, 1)) + w1(1);
    test_predict2 = exp(-test_radius.^2/2) * w2(2:size(w2, 1)) + w2(1);
    test_predict = (test_predict1 - test_predict2) > 0;
    train_errors(i) = sum(abs(ytrain1 - train_predict')) / train_length;
    test_errors(i) = sum(abs(ytest1 - test_predict')) / test_length;
end
hold on;
plot(lambda, train_errors, 'b');
plot(lambda, test_errors, 'r');
legend('train error', 'test error');
hold off;
end