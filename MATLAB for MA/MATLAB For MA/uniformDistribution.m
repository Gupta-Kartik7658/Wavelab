function U = uniformDistribution(N)

    arguments
        N = 2000000;
    end

    seed = uint32(mod(mod(convertTo(datetime,'excel'),1)*100000000,100000));
    a = 7^5;
    c = 5;
    m = double(intmax);
    
    X = lcg(seed,a,c,m,N);
    U = X/m;
    
    %subplot(2,1,1);
    %histogram(U,'Normalization','pdf');
    %xlabel('X');
    %ylabel('f_{x}(x)');
    %title('Standard Uniform Distribution PDF');

    %subplot(2,1,2);
    %histogram(U,'Normalization','cdf');
    %xlabel('X');
    %ylabel('F_{x}(x)');
    %title('Standard Uniform Distribution CDF');

end