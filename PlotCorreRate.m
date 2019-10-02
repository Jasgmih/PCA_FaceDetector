function [] = PlotCorreRate(fea,trainIdx,testIdx,gnd)

% the changes of accuracies,as the number of
% top principal components changes, 



k=[];
corr=[];
for i = 1:20
k = [k,i*5];
[eigVectors,W_train] = Training(fea,trainIdx,k(i));
[corrRate,W_test] = Testing(fea,trainIdx,testIdx,gnd,eigVectors,W_train);
corr = [corr,corrRate];
end
correct = [k',corr']
figure,plot(k,corr);
axis([0 100 0.58 0.85]);
xlabel('Number of top k principal components')
ylabel('accuracy')

end

