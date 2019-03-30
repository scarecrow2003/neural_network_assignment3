function w = q3b()
x = randn(800, 2);
s2 = sum(x.^2, 2);
xtrain = (x.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';
plot(xtrain(1, :), xtrain(2, :), 'r+');
width = 8;
w = rand(2, width * width);
t = 1000;
sigma0 = width / sqrt(2);
for i=1:2500
    sigma = sigma0 * exp(-i * log(sigma0) / t);
    eta = 0.1 * exp(-i / t); 
    for j=1:size(xtrain, 2)
        [~, idx] = min((w(1, :) - xtrain(1, j)).^2 + (w(2, :) - xtrain(2, j)).^2);
        idx_i = fix((idx-1) / width) + 1;
        idx_j = mod(idx-1, width) + 1;
        for k=1:size(w, 2)
            w_i = fix((k-1) / width) + 1;
            w_j = mod(k-1, width) + 1;
            d_square = (w_i-idx_i)^2 + (w_j-idx_j)^2;
            h = exp(-d_square / (2 * sigma ^2));
            w(:, k) = w(:, k) + eta * h * (xtrain(:, j) - w(:, k));
        end
    end
end
hold on;
for i=1:width
    plot(w(1, (i-1)*width+1:i*width), w(2, (i-1)*width+1:i*width), 'b-*');
    plot(w(1, i:8:(7*width+i)), w(2, i:8:(7*width+i)), 'b-*');
end
legend('train data', 'weight');
hold off;
end