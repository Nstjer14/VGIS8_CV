%% saveImages2

if exist('ForegroundImage')
    str = strcat('images/',TimeStamp,'.bmp');
    imwrite(ForegroundImage,str,'bmp')
end


h = figure(100);
%set(h, 'Visible', 'off');
hold on
plotCenterAndRotation
filename = strcat('images/',TimeStamp,'.bmp');
saveas(h,filename);
hold off;