function X = imageDistribution(path)
    img = imread(path);
    img = imresize(img,[512,512]);
    img = rgb2gray(img);
    X = img(:);
end

