function [V] = func_impulsenoise(V)

% 对每一列向量进行操作
for i = 1:size(V, 2)
    % 随机选择225个位置
    indices = randperm(size(V, 1), 225);

    % 在这些位置上随机置0或255
    V(indices, i) = randi([0, 255], 1, length(indices));
end

end 