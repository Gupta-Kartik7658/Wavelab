clear 
clc

mx = 3;
my = 5;
stdx = 0.25;
stdy = 0.5;

xlim = linspace(mx-(5*stdx),mx+(5*stdx),100);
ylim = linspace(my-(5*stdy),my+(5*stdy),100);


[x ,y] = meshgrid(xlim,ylim);

X = (x-mx)./stdx;
Y = (y-my)./stdy;

fxy = (1/(2*pi)) * exp(-(1/2) * (X.^2 + Y.^2));

Fxy = @(x,y)((1/(2*pi)) * exp(-(1/2) * (x.^2 + y.^2)));
cdf = zeros(size(X));

for i = 1:size(X)
    for j = 1:size(Y)
        cdf(i,j) = integral2(Fxy,mx-(5*stdx),X(1,i),my-(5*stdy),Y(j,1));
    end
end

figure(1)
scatter3(x(:),y(:),fxy(:));
title('Scatter PDF Plot for Bivariate Standard Normal Curve');
zlabel('f_{XY}(x,y)');
xlabel('X');
ylabel('Y');

figure(2)

surf(x,y,fxy);
title('Surface PDF Plot for Bivariate Standard Normal Curve');
zlabel('f_{XY}(x,y)');
xlabel('X');
ylabel('Y');

figure(3)

surf(x,y,cdf);
title('CDF Plot for Bivariate Standard Normal Curve');
zlabel('F_{XY}(x,y)');
xlabel('X');
ylabel('Y');



