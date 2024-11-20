function Cx = Covariance(X)
    [m, n] = size(X);
    mu = mean(X);
    X_c = X - mu;
    Cx = (1/(m-1)) .* (X_c' * X_c);
end