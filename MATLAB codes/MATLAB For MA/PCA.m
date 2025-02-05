function X_p = PCA(X,k)
    arguments
        X;
        k = size(X,2);
    end

    mu = mean(X);
    X_c = X - mean(X);
    Cx = Covariance(X);

    disp("Original Covariance Matrix");
    disp(Cx);

    [P, D] = eig(Cx);
    D = diag(D);
    
    [D, idx] = sort(D);
    P = P(:,idx);

    W = P(:,1:k);
    X_p = X_c * W;

    disp(["New Covariance Matrix for k = ",k]);
    disp(Covariance(X_p));
end