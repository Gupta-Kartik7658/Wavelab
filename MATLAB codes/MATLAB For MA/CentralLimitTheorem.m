clear 
clc

U = zeros(1,500000);
n = input("Number of Uniform distributions to add: ");
rows = (ceil(n/5));
for i=1:n
    x = uniformDistribution(500000);
    x_c = x - mean(x);
    U(1,:) = (U(1,:) + x_c);
    
    subplot(rows, 5,i);
    histogram(U./i,'Normalization','pdf');
    disp(["Standard Deviation of Plot with i = ", i," Distributions is: ", std(U./i)]);
end

%histogram(U,'Normalization','pdf');

