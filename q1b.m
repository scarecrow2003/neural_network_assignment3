function [w, cost] = q1b()
xtrain = (-1.6:0.08:1.6)';
fixed_points = 20;
m = xtrain(sort(randperm(size(xtrain, 1), fixed_points)));
n = randn(size(xtrain, 1), 1);
ytrain = 1.2 * sin(pi * xtrain) - cos(2.4 * pi * xtrain) + 0.3 * n;
radius = abs(xtrain - m');
dmax = max(abs(m - m'), [], 'all');
phi = [ones(size(xtrain, 1), 1) exp(-fixed_points * radius.^2/dmax^2)];
w = phi' * phi \ phi' * ytrain;
xtest = (-1.6:0.01:1.6)';
ypredict = exp(-fixed_points * (xtest-m').^2 / dmax^2) * w(2:size(w, 1)) + w(1);
ytest = 1.2 * sin(pi * xtest) - cos(2.4 * pi * xtest);
cost = sum((ytest - ypredict).^2) / 2;
hold on;
plot(xtrain, ytrain, 'g');
plot(xtest, ytest, 'b');
plot(xtest, ypredict, 'r');
legend('train data', 'test data', 'prediction');
hold off;
end