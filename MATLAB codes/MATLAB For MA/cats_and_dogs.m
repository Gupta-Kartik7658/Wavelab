clear all
clc

format longg
cats_path = dir(fullfile("C:\Users\hp\Desktop\MATLAB For MA\dog-cat-full-dataset-master\dog-cat-full-dataset-master\data\test\cats",'*jpg'));

number_of_images = 2500;
cats = zeros(512*512,number_of_images);

for i=1:number_of_images
    image = imread("C:\Users\hp\Desktop\MATLAB For MA\dog-cat-full-dataset-master\dog-cat-full-dataset-master\data\test\cats/"+cats_path(i).name);
    image = imresize(image,[512,512]);
    image = rgb2gray(image);
    
    cats(:,i) = image(:);
end


%Mean centering the data 
m_xd = mean(cats);
cats_centered = cats-m_xd;

% X store the column vectors of grayscaled version of every image
[mc ,nc] = size(cats_centered);



dogs_path = dir(fullfile("C:\Users\hp\Desktop\MATLAB For MA\dog-cat-full-dataset-master\dog-cat-full-dataset-master\data\test\dogs",'*jpg'));
dogs = zeros(512*512,number_of_images);

for i=1:number_of_images
    image = imread("C:\Users\hp\Desktop\MATLAB For MA\dog-cat-full-dataset-master\dog-cat-full-dataset-master\data\test\dogs/"+dogs_path(i).name);
    image = imresize(image,[512,512]);
    image = rgb2gray(image);
    dogs(:,i) = image(:);
end


%Mean centering the data 
m_xc = mean(dogs);
dogs_centered = dogs-m_xc;

% X store the column vectors of grayscaled version of every image
[md ,nd] = size(dogs_centered);


X = double(imageDistribution("C:\Users\hp\Desktop\dog.jpg"));
X = X-mean(X);
mx = mean(X);

Cx_dogs = (X'*dogs_centered) ./ (md-1);
Cx_cats = (X'*cats_centered) ./ (md-1);


if mean((Cx_cats)) > mean((Cx_dogs))
    disp("Cat");
else
    disp("Dog");
end


