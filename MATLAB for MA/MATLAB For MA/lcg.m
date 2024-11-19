function X = lcg(seed,a,c,m,N)
    arguments
        seed = 101;
        a = 7^5;
        c = 9;
        m = intmax;
        N = 2000000;
    end
    X = zeros(1,N);
    X(1) = double(seed); %Seed for LCG
    
    for i=2:N
        X(i) = mod((a*X(i-1) + c),m);
    end

end
