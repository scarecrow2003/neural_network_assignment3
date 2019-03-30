function [w, class] = q3c1()
data = load('Digits.mat');
idx = find(data.train_classlabel~=1 & data.train_classlabel~=2, 1000);
xtrain = data.train_data(:, idx);
ytrain = data.train_classlabel(idx);
width = 10;
w_flat = rand(784, width * width);
t = 1000;
sigma0 = width / sqrt(2);
for i=1:1000
    display("iteration: " + i);
    sigma = sigma0 * exp(-i * log(sigma0) / t);
    eta = 0.1 * exp(-i / t); 
    for j=1:size(xtrain, 2)
        [~, idx] = min(cellfun(@norm, num2cell((xtrain(:, j) - w_flat), 1)));
        idx_i = fix((idx-1) / width) + 1;
        idx_j = mod(idx-1, width) + 1;
        for k=1:size(w_flat, 2)
            w_i = fix((k-1) / width) + 1;
            w_j = mod(k-1, width) + 1;
            d_square = (w_i-idx_i)^2 + (w_j-idx_j)^2;
            h = exp(-d_square / (2 * sigma ^2));
            w_flat(:, k) = w_flat(:, k) + eta * h * (xtrain(:, j) - w_flat(:, k));
        end
    end
end
for i=1:100
    subplot(10,10,i), imshow(reshape(w_flat(:, i), 28, 28));
end
class_flat = zeros(1, width * width);
for i=1:width*width
    [~, idx] = min(cellfun(@norm, num2cell((w_flat(:, i) - xtrain), 1)));
    class_flat(i) = ytrain(idx);
end
w = reshape(num2cell(w_flat, 1), 10, 10);
class = reshape(class_flat, 10, 10);
end