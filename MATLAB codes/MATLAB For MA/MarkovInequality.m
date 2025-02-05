function p = MarkovInequality(X,alpha)
    Ex = mean(X);
    
    p = Ex/alpha;

    disp(["Markov's Upper bound for the following is: ",p]);
    cnt = 0;

    for i=1:numel(X)
        if(X(i)>=alpha)
            cnt = cnt +1;
        end
    end

    disp(["Actual probability for the following is: ",double(cnt)/numel(X)]);

end