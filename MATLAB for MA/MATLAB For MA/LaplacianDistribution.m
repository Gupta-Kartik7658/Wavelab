function L = LaplacianDistribution
    U1 = uniformDistribution();
    U2 = uniformDistribution();
    L = log(U1./U2);

    subplot(2,1,1);
    histogram(L,'Normalization','pdf');
    xlabel('X');
    ylabel('f_{x}(x)');
    title('Standard Laplacian Distribution PDF');

    subplot(2,1,2);
    histogram(L,'Normalization','cdf');
    xlabel('X');
    ylabel('F_{x}(x)');
    title('Standard Laplacian Distribution CDF');

end