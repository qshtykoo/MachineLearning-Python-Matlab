clear all
close all

load('lab1');
load('lab2');

for i = 1:6
    banana{i} = mat2gray(lab1{i});
    apple{i} = mat2gray(lab2{i});
end

figure;

for j = 1:6
     subplot(2,6,j)
     imshow(banana{j});
end

for j = 1:6
     subplot(2,6,j+6)
     imshow(apple{j});
end

