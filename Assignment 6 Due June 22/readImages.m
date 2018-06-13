function img = readImages(i, type)

if strcmp(type, 'b')
    fileDir = dir('banana\*.jpg');
    file_name = strcat('banana\',fileDir(i).name);
    img = imread(file_name);
elseif strcmp(type, 'a')
    fileDir = dir('apple\*.jpg');
    file_name = strcat('apple\',fileDir(i).name);
    img = imread(file_name);
end


    
