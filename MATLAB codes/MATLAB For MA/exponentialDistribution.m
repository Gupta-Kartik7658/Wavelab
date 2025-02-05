function E = exponentialDistribution(lambda,N)
    arguments
        lambda = 1;
        N = 2000000;
    end
    U = uniformDistribution();
    E = -log((1-U))/lambda;

    subplot(2,1,1);
    histogram(E,'Normalization','pdf');
    xlabel('X');
    ylabel('f_{x}(x)');
    title('Standard Exponential Distribution PDF');

    subplot(2,1,2);
    histogram(E,'Normalization','cdf');
    xlabel('X');
    ylabel('F_{x}(x)');
    title('Standard Exponential Distribution CDF');

end