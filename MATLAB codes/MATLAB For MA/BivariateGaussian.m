clear;
clc;

x = randperm(1000,50);
y = randperm(1000,50);

x = 10 .* x./max(x);
y = 10 .* y./max(y);

mx = mean(x);
my = mean(y);

X_ = [x' y'];
X_c = X_ - mean(X_);
[m ,n] = size(X_c);
Cx = (1/(m-1)) * (X_c' * X_c);
Cx_inv = inv(Cx);

[X, Y] = meshgrid(x, y);

fxy = zeros(numel(x));

for i = 1:50
    for j = 1:50
        fxy(i,j) = (1 / (2 * pi * sqrt(det(Cx)))) * exp(-0.5 * [X(i,j) - mx, Y(i,j) - my] * Cx_inv * [X(i,j) - mx, Y(i,j) - my]');
    end
end

scatter3(X(:), Y(:), fxy(:));
xlabel('X'); 
ylabel('Y'); 
zlabel('f_{xy}(x,y)');
title('Bivariate Gaussian PDF');
