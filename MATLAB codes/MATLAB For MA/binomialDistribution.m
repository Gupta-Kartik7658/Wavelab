function B = binomialDistribution(p,n)
    arguments
        p = 0.7;
        n = 200000;  
    end
        
    B = zeros(1,n);
    for i=1:n
        outcomes = rand(1,1000);
        B(1,i) = sum(outcomes<=p) - sum(outcomes>p);
    end

    subplot(2,1,1);
    histogram(B,'Normalization','probability');
    
    subplot(2,1,2);
    histogram(B,'Normalization','cdf');

    disp(["Mean",mean(B)]);
    disp(["Standard Deviation",std(B)]);

end