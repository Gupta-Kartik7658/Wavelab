function num = random()

    seed = uint32(mod(convertTo(datetime,'excel'),1)*100000);
    a = 7^5;
    c = 5;
    m = intmax;
    X = lcg(seed,a,c,m);
    
    num = X(numel(X));

end