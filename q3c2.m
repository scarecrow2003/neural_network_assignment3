function accuracy = q3c2()
[w, class] = q3c1();
w_flat = cell2mat(reshape(w, 1, 100));
class_flat = reshape(class, 1, 100);
data = load('Digits.mat');
idx = find(data.test_classlabel~=1 & data.test_classlabel~=2, 1000);
length = size(idx, 2);
xtest = data.test_data(:, idx);
ytest = data.test_classlabel(idx);
ypredict = zeros(1, length);
for i=1:length
    [~, idx] = min(cellfun(@norm, num2cell((xtest(:, i) - w_flat), 1)));
    ypredict(i) = class_flat(idx);
end
accuracy = size(find(ypredict - cast(ytest, 'double')==0), 2) / length;
end