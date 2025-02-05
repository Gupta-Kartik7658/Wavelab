function p = ChebyshevInequality(X,y)
    mx = mean(X);
    gamma = y - mx;
    sigma = var(X);
    p = var(X)/(gamma^2);
    Y = zeros(size(X));
    disp(["Chebyshev's Upper Bound is = ",p]);
    
    for i=1:numel(X)
        if(X(1,i)<= -gamma+mx || X(1,i)>= +gamma+mx)
            Y(1,i) = X(1,i);
        end
    end

    disp(["Actual Probabilty Value = ",double(sum(Y~=0))/numel(X)])
    
    Y = Y(Y~=0);
    subplot(2,1,1);
    hold on
    histogram(Y,100,'Normalization','probability');
    
end