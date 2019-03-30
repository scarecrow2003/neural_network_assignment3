function [train_predict, test_predict] = q2b()
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
fixed_center = 100;
m = xtrain(randperm(size(xtrain, 1), fixed_center));
radius = zeros(train_length, fixed_center);
for i=1:train_length
    for j=1:fixed_center
        radius(i, j) = norm(xtrain(:, i) - m(:, j));
    end
end
test_length = size(ytest, 2);
test_radius = zeros(test_length, fixed_center);
width = [0.1:0.1:1 2:1:9 10:5:50 100:100:1000 2000:1000:10000];
length = size(width, 2);
train_errors = zeros(1, length);
test_errors = zeros(1, length);
for i=1:length
    display("width: " + width(i));
    phi = [ones(train_length, 1) exp(-fixed_center * radius.^2/width(i)^2)];
    w1 = phi' * phi \ phi' * ytrain1';
    w2 = phi' * phi \ phi' * ytrain2';
    for j=1:test_length
        for k=1:fixed_center
            test_radius(j, k) = norm(xtest(:, j) - m(:, k));
        end
    end
    train_predict1 = exp(-fixed_center * radius.^2/width(i)^2) * w1(2:size(w1, 1)) + w1(1);
    train_predict2 = exp(-fixed_center * radius.^2/width(i)^2) * w2(2:size(w2, 1)) + w2(1);
    train_predict = (train_predict1 - train_predict2) > 0;
    test_predict1 = exp(-fixed_center * test_radius.^2/width(i)^2) * w1(2:size(w1, 1)) + w1(1);
    test_predict2 = exp(-fixed_center * test_radius.^2/width(i)^2) * w2(2:size(w2, 1)) + w2(1);
    test_predict = (test_predict1 - test_predict2) > 0;
    train_errors(i) = sum(abs(ytrain1 - train_predict')) / train_length;
    test_errors(i) = sum(abs(ytest1 - test_predict')) / test_length;
end
hold on;
plot(width, train_errors, 'b');
plot(width, test_errors, 'r');
legend('train error', 'test error');
hold off;
end