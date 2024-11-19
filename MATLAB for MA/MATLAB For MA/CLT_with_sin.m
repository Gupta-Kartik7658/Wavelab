U = zeros(1,500000);
n = input("Number of Uniform distributions to add: ");
rows = (ceil(n/3));
for i=1:n
    x = sin(2*pi*uniformDistribution(500000));
    x_c = x - mean(x);
    U(1,:) = U(1,:) + x_c;
    subplot(rows, 5,i);
    histogram(U,'Normalization','pdf');
end
%histogram(U,'Normalization','pdf');

