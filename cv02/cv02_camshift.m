clear all;
close all;
echo off;
clc;

vzor = imread('cv02_vzor_hrnecek.bmp', 'bmp');
%figure
%imshow(uint8(vzor))
si = size (vzor);
vyska2 = si(1) / 2; 
sirka2 = si(2) / 2;
hsv1 = rgb2hsv(vzor);
h = hsv1(:, :, 1) * 255;
hist = imhist(uint8(h));
hist = hist / max(hist);


v = VideoReader('cv02_hrnecek.mp4');
videoIM = read(v,1);
% figure
% imshow(uint8(videoIM))

id = 1;
idv = 1;
while (1)
    try
        videoIM = read(v, id);
    catch
        break;
    end
     if id == 1
         si1 = size (videoIM);
         x1 = 1;
         x2 = si1(2);
         y1 = 1;
         y2 = si1(1);
     end
     id = id + 1;
     hsv = rgb2hsv(videoIM);
     P = hist(uint8(hsv(:,:,1) *255) + 1);
     %vypocet teziste
     xt = 0;
     yt = 0;
     for x = x1:x2
         for y = y1:y2
             xt = xt + x*P(y, x);
             yt = yt + y*P(y, x);
         end
     end
     sumT = sum(sum(P(y1:y2, x1:x2)));
     xt = xt / sumT;
     yt = yt / sumT;
     x1 = floor(xt - sirka2);
     x2 = floor(xt + sirka2);
     y1 = floor(yt - vyska2);
     y2 = floor(yt + vyska2);
     %zobrazeni
     imshow(videoIM);
     hold on;
     rectangle('position',[x1 y1 sirka2*2 vyska2*2], 'LineWidth', 2, 'Edgecolor', 'g')
     hold off;
     
     F(idv) = getframe(gcf);
     idv = idv + 1;
     %pause(0.01);
end


