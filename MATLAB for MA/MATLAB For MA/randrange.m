function num = randrange(p,q)

    U = uniformDistribution();
    close all;
    
    R = p + (q-p)*U;
    
    num = double(R(numel(R)));

end