function wrenchVSstop()
clear all

trainTestRatio = 0.5;
lr = 0.0005;
f = @logsig;
gradf = @(x) f(x).*(1-f(x));

[xTrain, dTrain, xTest, dTest, imgTest] = Load('laptop', 'stop_sign', trainTestRatio);
[w, E] = OfflineLearning(xTrain, dTrain, f, gradf, lr, @Stop);

y = Predict(xTest, f, w);
y = OutputToClass(y, max(dTest(:)));
Draw(y, dTest, imgTest);

function yy = OutputToClass(y, number)
yy= y > number/2;

function Draw(predicted, actual, X)
c = ["laptop", "stop sign"];
close all;
nrows = 6;
ncols = 6;
predicted = predicted + 1;
actual = actual +1;

for i = 1:nrows-1
    for j = 1:ncols-1
        k = (i-1)*ncols + j;
        subplot(nrows, ncols, k);
        img = uint8(reshape(X(k,:), 64, 64));
        imshow(img);
        xlabel(c(predicted(k)));
    end
end
set(gcf, 'Position', [50,211,560,690]);

figure
confusionchart(c(actual), c(predicted));
set(gcf, 'Position', [50,0,560,136]);


function [xTrain, dTrain, xTest, dTest, imgTest] = Load(folder1, folder2, trainTestRatio)
[img0, N0] = LoadFolder(folder1);
[img1, N1] = LoadFolder(folder2);
d = [zeros(N0,1); ones(N1,1)];
img = [img0; img1];
N = N0 + N1;
p = randperm(N);
d = d(p);
X = zscore(img(p,:));
n = round(N*trainTestRatio);
xTrain = X(1:n,:);
xTest = X(n+1:end, :);
dTrain = d(1:n);
dTest = d(n+1:end);
imgTest = img(p(n+1:end), :);

function [img, N] = LoadFolder(folder)
files = dir([folder '/*.jpg']);
N = length(files);
img = [];
for i = 1:N
   fprintf('loading %s\n', files(i).name);
   img_i = imread([folder '/' files(i).name]);
   img_i = imresize(img_i, [64,64]);
   if (size(img_i, 3) == 3), img_i = rgb2gray(img_i); end
   img(i, :) = double(img_i(:))';
end

function z = Stop(E, epoch)
if (epoch > 10000)
    z = true;
    return;
end
if (length(E)<10)
    z = false;
    return;
end

if E(end-9) < E(end) || E(end-9) - E(end)< 1e-3
    z = true;
    return
end

z = false;

function y = Predict(x,f,w)
y = zeros(size(x,1), size(w,1));

for i = 1:size(x,1)
    r = x(i,:);
    q=r+w(i,:);
   z = f((x(i,:)+w(i, :)));
   y(i,:) = f(q); 
end

function [w, E] = OfflineLearning(x, d, f, gradf, lr, stop)
[~, n] = size(x);
w = randn(n,size(d,2));
epoch = 0;
while true
    v = x * w;
    y = f(v);
    e = y - d; % using square loss function
    g = x' * (e .* gradf(v)); %
    w = w - lr * g;
    E = sum(e(:).^2);
    if stop(E, epoch), break; end
    epoch = epoch + 1;
end