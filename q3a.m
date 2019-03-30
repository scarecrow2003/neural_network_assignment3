function w = q3a()
length = 40;
x = linspace(-pi, pi, length);
xtrain = sinc(x);
w = rand(length, 1);
t = 1000;
for i=1:2500
    sigma = (length / 2) * exp(-i * log(length / 2) / t);
    eta = 0.1 * exp(-i / t); 
    for j=1:length
        [~, idx] = min(abs(w - xtrain(j)));
        for k=1:length
            h = exp(-(k - idx)^2 / (2 * sigma ^2));
            w(k) = w(k) + eta * h * (xtrain(idx) - w(k));
        end
    end
end
hold on;
plot(x, xtrain, 'b');
plot(x, w, 'r-*');
legend('train data', 'weight');
hold off;
end