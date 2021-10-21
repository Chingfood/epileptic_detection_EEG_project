load('data1.mat')
% columns 1 to end-1 are input features, the last (end) column contain
% labels
X = data(:,1:end-1);
y = data(:,end); % the last column is the class labels

figure;
gscatter(X(:,1),X(:,2),y); % creates a scatter plot and uses the class labels to color the data points

%% Creating training and test set 
cvp = cvpartition(y, 'Holdout', 0.25); % partitions the data into 25% for testing and 75% for training
training = cvp.training; %returns the logical vector of training indices
test = cvp.test; %returns the logical vector of test indices

train_X = X(training,:); % form the training data
train_y = y(training); 
    
test_X = X(test,:); % form the testing data
test_y = y(test);    

%% classifies the data using a linear SVM model
model = fitcsvm(train_X, train_y); % develop the SVM model on training data

train_pred_y = predict(model, train_X); % use the trained model to classify the training data
train_acc = sum(train_y == train_pred_y)/length(train_y); % find the classifiction accuracy on the training data          
test_pred_y = predict(model, test_X); % use the trained model to classify the testing data 
test_acc = sum(test_y == test_pred_y)/length(test_y) % find the classifiction accuracy on the training data
svm_plot_classes(X,y,model); % plots the data and the class boundaries

%% classifies the data using a RBF (Gaussian Radial Basis Function)kernel SVM model, kernel_scale = 1
% kernel scale specifies how far the influence of a single training example reaches
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 1); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X);
train_acc = sum(train_y == train_pred_y)/length(train_y); 
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y)
svm_plot_classes(X,y,model); 

%% classifies the data using a RBF (Gaussian Radial Basis Function)kernel SVM model, kernel_scale = 0.1
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 0.1); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y) 
svm_plot_classes(train_X,train_y,model);
svm_plot_classes(test_X,test_y,model);
%% classifies the data using a RBF (Gaussian Radial Basis Function)kernel SVM model, kernel_scale = 100
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 100); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);        
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y) 
svm_plot_classes(X,y,model); % cannot capture the complexity or “shape” of the data.

%% classifies the data using a RBF (Gaussian Radial Basis Function)kernel SVM model, kernel_scale = 1, 'BoxConstraint' parameter, C = 100
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 1, 'BoxConstraint', 100); 

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);     
test_pred_y = predict(model, test_X);
test_acc = sum(test_y == test_pred_y)/length(test_y) 
svm_plot_classes(X,y,model);

%% classifies the data using a RBF (Gaussian Radial Basis Function)kernel SVM model, kernel_scale = 1, 'BoxConstraint' parameter, C = 0.1
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 1, 'BoxConstraint', 0.1);

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);       
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y) 
svm_plot_classes(X,y,model); 

%% choosing the best 'KernelScale' (kernel_scale) and 'BoxConstraint' (C) values
% by performing grid search to find the best combination of sigma (p_best) 
% and C (c_best) parameters, trains an SVM model using these parameters and evaluates it on the test data.
kernel_scales = [0.01 0.1 1 5 10 50 100 500 1000 5000];
Cs = [0.01 0.1 1 5 10 50 100 500 1000 5000];
[kernel_scale_best, C_best] = svm_hyperparameter_tuning(train_X,train_y,'rbf',kernel_scales,Cs);
% Use the best parameters to train a classifier with all the data
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', kernel_scale_best, 'BoxConstraint', C_best);
train_pred_y = predict(model, train_X);
train_acc = sum(train_y == train_pred_y)/length(train_y);        
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y)
svm_plot_classes(X,y,model); 

%% Naive Bayes: you might need to install Statistics and Machine Learning Toolbox
model = fitcnb(train_X, train_y);
train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);        
test_pred_y = predict(model, test_X);
test_acc = sum(test_y == test_pred_y)/length(test_y) 

%% Random forest
model = TreeBagger(10,train_X, train_y,'MinLeafSize',7);
train_pred_y = predict(model, train_X); 
train_pred_y = cellfun(@(x)str2double(x), train_pred_y);
train_acc = sum(train_y == train_pred_y)/length(train_y); 
test_pred_y = predict(model, test_X); 
test_pred_y = cellfun(@(x)str2double(x), test_pred_y);
test_acc = sum(test_y == test_pred_y)/length(test_y)
tree_plot_classes(X,y,model); 
close all

%% compute AUROC
% [X_cor,Y_cor,T,AUC] = perfcurve(labels,scores,posclass)
[~,score] = predict(model, test_X);
[x_cor,y_cor,T,AUC] = perfcurve(test_y,score(:,2),2);
figure;plot(x_cor,y_cor)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Random Forest') % does this look correct?

%% compute F-score: F-score = 2*(precision*recall)/(precision+recall)
% precision = TP/(TP+FP)
precision = sum(test_y==1 & test_pred_y==1) / sum(test_pred_y==1)
% recall = TP/(TP+FN)
recall = sum(test_y==1 & test_pred_y==1) / sum(test_y==1)
F = 2*(precision*recall)/(precision+recall)

%% cross validation
cvp = cvpartition(y,'KFold',5);
AUC_all = zeros(cvp.NumTestSets,1);
for i = 1:cvp.NumTestSets
    disp(['Fold ',num2str(i)])
    training = cvp.training(i);
    test = cvp.test(i);
    
    train_X = X(training,:); % form the training data
    train_y = y(training); 
    
    test_X = X(test,:); % form the testing data
    test_y = y(test);    

     model = fitctree(train_X,train_y);
     [~, score] = predict(model, test_X);
     
     [~,~,~,AUC_all(i)] = perfcurve(test_y,score(:,2),2);
end
mean(AUC_all)
figure; boxplot(AUC_all)

clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feature selection using information gain
% Load in data
% https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
% Downloaded from:
% https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
% Missing data removed
% data: features, with random noise columns added
% labels: the class labels
load('noisy_breast_cancer_data.mat');  %label: (2 for benign, 4 for malignant)

%% Partition the data as 75% training, 25% testing
cvp = cvpartition(labels, 'Holdout', 0.25);
training = cvp.training;
test = cvp.test;

train_X = data(training,:); % form the training data
train_y = labels(training); 
    
test_X = data(test,:); % form the testing data
test_y = labels(test);    

%% Use all features to generate a SVM model
model = fitcsvm(train_X, train_y);
test_pred_y = predict(model, test_X); 
test_acc_tree_SVM = sum(test_y == test_pred_y)/length(test_y);

%% Calculate the informatin gain for each feature
ig = infogain(data,labels);
% Select only those features with an information gain above 0.1
selected_features = ig > 0.1;

train_X = data(training,selected_features);
test_X = data(test,selected_features);

%% Re-run the SVM algorithm but only using the selected features 
model = fitcsvm(train_X, train_y);
test_pred_y = predict(model, test_X); 
test_acc_clean_tree = sum(test_y == test_pred_y)/length(test_y);

clear all
close all










