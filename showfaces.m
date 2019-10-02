function [image] = showfaces(faceMatr)
    image=[];
    Faces = reshape(faceMatr,32,320,[]);

    for i = 1:size(Faces,3)
        image = [image;Faces(:,:,i)];
    end
    figure,imshow(image,[]);
end

