function center = twomeancluster(data)
length = size(data, 2);
center = data(:, [1, 2]);
idx = zeros(1, length);
while true
    for i=1:length
        distance1 = norm(data(:, i) - center(:, 1));
        distance2 = norm(data(:, i) - center(:, 2));
        if (distance1 < distance2)
            idx(i) = 1;
        else
            idx(i) = 2;
        end
    end
    new_center(:, 1) = mean(data(:, idx==1), 2);
    new_center(:, 2) = mean(data(:, idx==2), 2);
    center_distance1 = norm(center(:, 1) - new_center(:, 1));
    center_distance2 = norm(center(:, 2) - new_center(:, 2));
    display("center distance 1: " + center_distance1);
    display("center distance 2: " + center_distance2);
    if center_distance1 < 0.01 && center_distance2 < 0.01
        break;
    else
        center = new_center;
    end
end
end