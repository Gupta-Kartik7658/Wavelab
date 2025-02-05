function Z = gaussianDistribution(mu,sigma,N)
    arguments
        mu = 0;
        sigma = 1;
        N = 2000000;
    end
    
    U1 = uniformDistribution(N);
    U2 = uniformDistribution(N);
    Z1 = (-2.*log(U1)).^0.5;
    Z2 = cos(2*pi*U2);
    Z = mu + sigma.*(Z1.*Z2);

    subplot(2,1,1);
    histogram(Z,'Normalization','pdf');
    xlabel('X');
    ylabel('f_{x}(x)');
    title('Standard Guassian Distribution PDF');

    subplot(2,1,2);
    histogram(Z,'Normalization','cdf');
    xlabel('X');
    ylabel('F_{x}(x)');
    title('Standard Gaussian Distribution CDF');
end