function [train_error, test_error] = q2c()
data = load('mnist_m.mat');
xtrain = data.train_data;
dimension = size(xtrain, 1);
center = twomeancluster(xtrain);
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
radius = zeros(train_length, 2);
for i=1:train_length
    for j=1:2
        radius(i, j) = norm(xtrain(:, i) - center(:, j));
    end
end
test_length = size(ytest, 2);
test_radius = zeros(test_length, 2);
width = norm(center(:, 1) - center(:, 2));
phi = [ones(train_length, 1) exp(-2 * radius.^2/width^2)];
w1 = phi' * phi \ phi' * ytrain1';
w2 = phi' * phi \ phi' * ytrain2';
for j=1:test_length
    for k=1:2
        test_radius(j, k) = norm(xtest(:, j) - center(:, k));
    end
end
train_predict1 = exp(-2 * radius.^2/width^2) * w1(2:size(w1, 1)) + w1(1);
train_predict2 = exp(-2 * radius.^2/width^2) * w2(2:size(w2, 1)) + w2(1);
train_predict = (train_predict1 - train_predict2) > 0;
test_predict1 = exp(-2 * test_radius.^2/width^2) * w1(2:size(w1, 1)) + w1(1);
test_predict2 = exp(-2 * test_radius.^2/width^2) * w2(2:size(w2, 1)) + w2(1);
test_predict = (test_predict1 - test_predict2) > 0;
train_error = sum(abs(ytrain1 - train_predict')) / train_length;
test_error = sum(abs(ytest1 - test_predict')) / test_length;
display("train error: " + train_error);
display("test error: " + test_error);
classmean = zeros(dimension, 10);
for i=1:10
    classmean(:, i) = mean(xtrain(:, find(data.train_classlabel==i-1)), 2);
end
subplot(2, 10, 1), imshow(reshape(center(:, 1), 28, 28));
label1mean = mean(classmean(:, [2, 5]), 2);
subplot(2, 10, 2), imshow(reshape(label1mean, 28, 28));
subplot(2, 10, 3), imshow(reshape(classmean(:, 2), 28, 28));
subplot(2, 10, 4), imshow(reshape(classmean(:, 5), 28, 28));

subplot(2, 10, 11), imshow(reshape(center(:, 2), 28, 28));
label2mean = mean(classmean(:, [1, 3, 4, 6, 7, 8, 9, 10]), 2);
subplot(2, 10, 12), imshow(reshape(label2mean, 28, 28));
subplot(2, 10, 13), imshow(reshape(classmean(:, 1), 28, 28));
subplot(2, 10, 14), imshow(reshape(classmean(:, 3), 28, 28));
subplot(2, 10, 15), imshow(reshape(classmean(:, 4), 28, 28));
subplot(2, 10, 16), imshow(reshape(classmean(:, 6), 28, 28));
subplot(2, 10, 17), imshow(reshape(classmean(:, 7), 28, 28));
subplot(2, 10, 18), imshow(reshape(classmean(:, 8), 28, 28));
subplot(2, 10, 19), imshow(reshape(classmean(:, 9), 28, 28));
subplot(2, 10, 20), imshow(reshape(classmean(:, 10), 28, 28));
end