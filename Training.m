function [eigVectors,W_train] = Training(fea,trainIdx,k)

if (nargin<3)
    k = 40;
end

% showfaces(fea');
trainFaces = fea(trainIdx,:)';
% showfaces(trainFaces);
% title('Training faces');

mu=mean(trainFaces,2);
mu_face = reshape(mu,32,[]);
figure,imshow(mu_face, [0 255]);
title('Mean face');

%% get eigenvectors
X_train = trainFaces - repmat(mu,1,size(trainFaces,2));
[x y] = eig(X_train'*X_train);
eigVectors= X_train*x;

%% top k eigVectors

eigVectors = eigVectors(:,size(eigVectors,2)-k+1:end);

%% showing the eigenvectors faces
eigFaces=eigVectors+repmat(mu,1,size(eigVectors,2));

if mod(k,10)==0
showfaces(fliplr(eigFaces));
title('Top K principal components');
end

%% projection
W_train = eigVectors'*X_train; % every columns is a weight vector of corresponding face

%% reconstructed training face
TraFacesReconst = repmat(mu,1,size(W_train,2))+eigVectors*W_train;
showfaces(TraFacesReconst);
title('Reconstruction training faces');

end

