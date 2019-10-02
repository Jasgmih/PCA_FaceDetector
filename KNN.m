function [class] = KNN(trainData,testData,labels,K)
% Every row in trainData is a point
% TestData is a Row vector containing one point
% labels is a vector labeling every point in trainData

distance = pdist2(trainData,testData);
[distance_new index] = sort(distance);

nearestNer = labels(find(distance < distance_new(K)));

table = tabulate(nearestNer);
class = table(find(table(:,2) == max(table(:,2))),1);

end

