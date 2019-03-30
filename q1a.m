function [w, cost] = q1a()
xtrain = (-1.6:0.08:1.6)';
n = randn(size(xtrain, 1), 1);
ytrain = 1.2 * sin(pi * xtrain) - cos(2.4 * pi * xtrain) + 0.3 * n;
radius = abs(xtrain' - xtrain);
phi = exp(-radius.^2/(2 * 0.1^2));
w = inv(phi) * ytrain;

xtest = (-1.6:0.01:1.6)';
ypredict = exp(-(abs(xtest - xtrain')).^2/0.02) * w;
ytest = 1.2 * sin(pi * xtest) - cos(2.4 * pi * xtest);
cost = sum((ytest - ypredict).^2) / 2;
hold on;
plot(xtrain, ytrain, 'g');
plot(xtest, ytest, 'b');
plot(xtest, ypredict, 'r');
legend('train data', 'test data', 'prediction');
hold off;
end