function [corrRate,W_test] = Testing(fea,trainIdx,testIdx,gnd,eigVectors,W_train,numbNearest)

if (nargin<7)
    numbNearest = 3;
end

TrainLabel = gnd(trainIdx);

TestFaces = fea(testIdx,:)';
TestLabel = gnd(testIdx);
% showfaces(TestFaces);
% title('Testing faces');

mu=mean(TestFaces,2);
mu_face = reshape(mu,32,[]);
% figure,imshow(mu_face, [0 255]);

%% Projection
X_test = TestFaces - repmat(mu,1,size(TestFaces,2));
W_test = eigVectors'*X_test;

%% correct rate

numberUncorr = 0;
for i = 1:size(W_test,2)
    class = KNN (W_train',W_test(:,i)',TrainLabel,numbNearest);
    if isempty(find(class == gnd(testIdx(i)))) 
        numberUncorr = numberUncorr+1;
    end
end

corrRate = 1-numberUncorr/size(W_test,2)
end

