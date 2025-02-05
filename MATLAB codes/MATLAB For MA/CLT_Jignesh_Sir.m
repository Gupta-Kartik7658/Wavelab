N = 25;
k = 10000;


for i = 1:N
   
        u(i,:) = rand(1,k); % you can change the distribution here and shall get the same results
end

yk = zeros(1,k);

yk(1,:) = (u(1,:)+u(1,:))-mean(u(1)); % mean subtracted

for i = 2:N
    
    yk(i,:) = (yk(i-1,:)+ u(i,:)) -mean(u(i)) ; % mean subtracted
end


for i = 1:N
    subplot(5,5,i);hist(yk(i,:)); 
end

std(yk)