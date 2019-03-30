function [w, cost] = q1c()
xtrain = (-1.6:0.08:1.6)';
n = randn(size(xtrain, 1), 1);
ytrain = 1.2 * sin(pi * xtrain) - cos(2.4 * pi * xtrain) + 0.3 * n;
radius = abs(xtrain' - xtrain);
phi = [ones(size(xtrain, 1), 1) exp(-radius.^2/(2 * 0.1^2))];
xtest = (-1.6:0.01:1.6)';
ytest = 1.2 * sin(pi * xtest) - cos(2.4 * pi * xtest);
lambda = [0:0.05:0.9 1:1:10 15:5:50];
length = size(lambda, 2);
cost = zeros(1, length);
for i=1:length
    w = (phi' * phi + lambda(i) * eye(size(phi, 2))) \ phi' * ytrain;
    ypredict = exp(-(xtest - xtrain').^2/0.02) * w(2:size(w, 1)) + w(1);
    cost(i) = sum((ypredict - ytest).^2)/2;
    if lambda(i) == 0.55
        ypredictmin = ypredict;
    end
end
% section a, plot cost versus lambda, comment section b when plotting
% section a
%hold on;
%plot(lambda, cost);
%legend('cost');
%hold off;
   
% section b, plot train data, test data and predicted test data, comment 
% section a when plotting section b
hold on;
plot(xtrain, ytrain, 'g');
plot(xtest, ytest, 'b');
plot(xtest, ypredictmin, 'r');
legend('train data', 'test data', 'prediction');
hold off;
cost = sum((ytest - ypredictmin).^2) / 2;
end