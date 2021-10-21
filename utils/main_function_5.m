
%% 1.a
load('data3.mat')

figure;
gscatter(X(:,1),X(:,2),y);
xlabel('x value');
ylabel('y value');

%% 1.b
cvp = cvpartition(y, 'Holdout', 0.25); % partitions the data into 25% for testing and 75% for training
training = cvp.training; %returns the logical vector of training indices
test = cvp.test; %returns the logical vector of test indices

train_X = X(training,:); % form the training data
train_y = y(training); 
    
test_X = X(test,:); % form the testing data
test_y = y(test);    

model = fitcsvm(train_X, train_y); % develop the SVM model on training data

train_pred_y = predict(model, train_X); % use the trained model to classify the training data
train_acc = sum(train_y == train_pred_y)/length(train_y); % find the classifiction accuracy on the training data          
test_pred_y = predict(model, test_X); % use the trained model to classify the testing data 
test_acc = sum(test_y == test_pred_y)/length(test_y); % find the classifiction accuracy on the training data
svm_plot_classes(X,y,model); 
xlabel('x value')
ylabel('y value')
title('Linear SVM')

%% 1.c

model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 1*sqrt(2)); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X);
train_acc = sum(train_y == train_pred_y)/length(train_y); 
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y)
svm_plot_classes(X,y,model); 

xlabel('x value')
ylabel('y value')
title('SVM RBF kernal sigma =1')

%% 1.d
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 0.1*sqrt(2)); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y)
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y) 
svm_plot_classes(X,y,model);
xlabel('x value')
ylabel('y value')
title('SVM RBF sigma = 0.1')

%classifies the data using a RBF (Gaussian Radial Basis Function)kernel SVM model, kernel_scale = 100
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 10*sqrt(2)); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X); 
train_acc = sum(train_y == train_pred_y)/length(train_y);  
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y) ;
svm_plot_classes(X,y,model); % cannot capture the complexity or “shape” of the data.
xlabel('x value')
ylabel('y value')
title('SVM RBF sigma = 10')


%% 1.e
Cs = [0.1 1 10 100 1000];
test_accs = zeros(length(Cs),1);
train_accs=zeros(length(Cs),1);
for j = 1:length(Cs)
        model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', sqrt(2), 'BoxConstraint', Cs(j));
        train_pred_y = predict(model, train_X); 
        train_accs(j) = sum(train_y == train_pred_y)/length(train_y)  ;
        test_pred_y = predict(model, test_X); 
        test_accs(j) = sum(test_y == test_pred_y)/length(test_y) ;
        svm_plot_classes(X,y,model); % cannot capture the com
end

close all;
%% 1.f
test_accs = zeros(10,1);
train_accs=zeros(10,1);
for i = 1:10
model = fitcsvm(train_X, train_y, 'KernelFunction', 'polynomial', 'PolynomialOrder', i); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X); 
train_accs(i) = sum(train_y == train_pred_y)/length(train_y);  
test_pred_y = predict(model, test_X); 
test_accs(i) = sum(test_y == test_pred_y)/length(test_y); 
svm_plot_classes(X,y,model);  % cannot capture the complexity or “shape” of the data.
xlabel('x value')
ylabel('y value')
end

close all;

%% 1.g
% see 1.f

%% 1.h
kernel_scales = [0.1 1 5 10 50 100 500 1000] .* sqrt(2);
Cs =  [0.1 1 5 10 50 100 500 1000];
[kernel_scale_best,C_best,train_X_new, train_y_new, tune_up_X, tune_up_y] = svm_hyperparameter_tuning_new(train_X,train_y,'rbf',kernel_scales,Cs);
model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', kernel_scale_best,  'BoxConstraint', C_best); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X_new); 
train_acc = sum(train_y_new == train_pred_y)/length(train_y_new);  
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y) ;
tune_up_pred_y = predict(model, tune_up_X); 
tune_up_acc = sum(tune_up_y == tune_up_pred_y)/length(tune_up_y) ;

svm_plot_classes(train_X_new,train_y_new,model); % cannot capture the complexity or “shape” of the data.
xlabel('x value')
ylabel('y value')
title('SVM RBF training sigma = 1 boxrestraint = 10 ')

svm_plot_classes(tune_up_X,tune_up_y,model); % cannot capture the complexity or “shape” of the data.
xlabel('x value')
ylabel('y value')
title('SVM RBF validation sigma = 1 boxrestraint = 10 ')

svm_plot_classes(test_X,test_y,model); % cannot capture the complexity or “shape” of the data.
xlabel('x value')
ylabel('y value')
title('SVM RBF test sigma= 1 boxrestraint = 10 ')
%% 1.i
%test
TP_test = sum(test_y==2 & test_pred_y==2)
FP_test = sum(test_y==1 & test_pred_y ==2)
TN_test = sum(test_y==1 & test_pred_y==1)
FN_test = sum(test_y==2 & test_pred_y==1)
TP_test_rate = TP_test/(TP_test+FN_test)
FP_test_rate = FP_test/(FP_test+TN_test)
TN_test_rate = TN_test/(TN_test +FP_test)
FN_test_rate = FN_test/(FN_test + TP_test)
% precision = TP/(TP+FP)
precision = sum(test_y==1 & test_pred_y==1) / sum(test_pred_y==1)
% recall = TP/(TP+FN)
recall = sum(test_y==1 & test_pred_y==1) / sum(test_y==1)
F = 2*(precision*recall)/(precision+recall)

%train
TP_train = sum(train_y_new==2 & train_pred_y==2)
FP_train = sum(train_y_new==1 & train_pred_y ==2)
TN_train = sum(train_y_new==1 & train_pred_y==1)
FN_train = sum(train_y_new==2 & train_pred_y==1)
TP_train_rate = TP_train/(TP_train+FN_train)
FP_train_rate = FP_train/(FP_train+TN_train)
TN_train_rate = TN_train/(TN_train +FP_train)
FN_train_rate = FN_train/(FN_train + TP_train)
% precision = TP/(TP+FP)
precision_train = sum(train_y_new==1 & train_pred_y==1) / sum(train_pred_y==1)
% recall = TP/(TP+FN)
recall_train = sum(train_y_new==1 & train_pred_y==1) / sum(train_y_new==1)
F_train = 2*(precision_train*recall_train)/(precision_train+recall_train)

%validation
TP_val = sum(tune_up_y==2 & tune_up_pred_y==2)
FP_val = sum(tune_up_y==1 & tune_up_pred_y ==2)
TN_val = sum(tune_up_y==1 & tune_up_pred_y ==1)
FN_val = sum(tune_up_y==2 & tune_up_pred_y ==1)
TP_val_rate = TP_val/(TP_val+FN_val)
FP_val_rate = FP_val/(FP_val+TN_val)
TN_val_rate = TN_val/(TN_val +FP_val)
FN_val_rate = FN_val/(FN_val + TP_val)
% precision = TP/(TP+FP)
precision_val = sum(tune_up_y==1 & tune_up_pred_y==1) / sum(tune_up_pred_y==1)
% recall = TP/(TP+FN)
recall_val = sum(tune_up_y==1 & tune_up_pred_y==1) / sum(tune_up_y==1)
F_val = 2*(precision_val*recall_val)/(precision_val+recall_val)

%% 2

cvp = cvpartition(y,'KFold',10);
AUC_svm = zeros(cvp.NumTestSets,1);
F_svm= zeros(cvp.NumTestSets,1);
for i = 1:cvp.NumTestSets
    disp(['Fold ',num2str(i)])
    training = cvp.training(i);
    test = cvp.test(i);
    
    train_X = X(training,:); % form the training data
    train_y = y(training); 
    
    test_X = X(test,:); % form the testing data
    test_y = y(test);    

     
     model = fitcsvm(train_X, train_y, 'KernelFunction', 'rbf', 'KernelScale', 5*sqrt(2),  'BoxConstraint', 5); % 'BoxConstraint' parameter, C, is set to 1 as a default

     [test_pred_y, score]= predict(model, test_X);
     precision = sum(test_y==1 & test_pred_y==1) / sum(test_pred_y==1);
     recall = sum(test_y==1 & test_pred_y==1) / sum(test_y==1);
     F_svm(i) = 2*(precision*recall)/(precision+recall);
     [~,~,~,AUC_svm(i)] = perfcurve(test_y,score(:,2),2);
end


%% 3
AUC_randfore = zeros(cvp.NumTestSets,1);
F_randfore =zeros(cvp.NumTestSets,1);
for i = 1:cvp.NumTestSets
    disp(['Fold ',num2str(i)])
    training = cvp.training(i);
    test = cvp.test(i);
    
    train_X = X(training,:); % form the training data
    train_y = y(training); 
    
    test_X = X(test,:); % form the testing data
    test_y = y(test);    
    model = TreeBagger(20,train_X, train_y,'MinLeafSize',10);
    [test_pred_y, score] = predict(model, test_X); 
    test_pred_y = cellfun(@(x)str2double(x), test_pred_y);
    
    tree_plot_classes(X,y,model); 
    xlabel('x value')
    ylabel('y value')
    title('random forest training MinLeafSize = 10, 20 decision trees ')
    precision = sum(test_y==1 & test_pred_y==1) / sum(test_pred_y==1);
    recall = sum(test_y==1 & test_pred_y==1) / sum(test_y==1);
    F_randfore(i) = 2*(precision*recall)/(precision+recall);
    [~,~,~,AUC_randfore(i)] = perfcurve(test_y,score(:,2),2);
end
close all;

%% 4
AUC_knn = zeros(cvp.NumTestSets,1);
F_knn = zeros(cvp.NumTestSets,1);
for i = 1:cvp.NumTestSets
    disp(['Fold ',num2str(i)])
    training = cvp.training(i);
    test = cvp.test(i);
    
    train_X = X(training,:); % form the training data
    train_y = y(training); 
    
    test_X = X(test,:); % form the testing data
    test_y = y(test);    
    model = fitcknn(train_X, train_y,'NumNeighbors',10);
    [test_pred_y, score] = predict(model, test_X); 
    nb_and_knn_plot_classes(X,y,model); 
    xlabel('x value')
    ylabel('y value')
    title('KNN training NumNeighbors = 10 ')
    precision = sum(test_y==1 & test_pred_y==1) / sum(test_pred_y==1);
    recall = sum(test_y==1 & test_pred_y==1) / sum(test_y==1);
    F_knn(i) = 2*(precision*recall)/(precision+recall);
    [~,~,~,AUC_knn(i)] = perfcurve(test_y,score(:,2),2);
end

close all;



%% 5
AUC_nb = zeros(cvp.NumTestSets,1);
F_nb = zeros(cvp.NumTestSets,1);

for i = 1:cvp.NumTestSets
    disp(['Fold ',num2str(i)])
    training = cvp.training(i);
    test = cvp.test(i);
    
    train_X = X(training,:); % form the training data
    train_y = y(training); 
    
    test_X = X(test,:); % form the testing data
    test_y = y(test);    

     
     model = fitcnb(train_X, train_y);
     [test_pred_y, score]= predict(model, test_X);
     nb_and_knn_plot_classes(X,y,model); 
     xlabel('x value')
     ylabel('y value')
     title('Naive Bayes training')
     precision = sum(test_y==1 & test_pred_y==1) / sum(test_pred_y==1);
     recall = sum(test_y==1 & test_pred_y==1) / sum(test_y==1);
     F_nb(i) = 2*(precision*recall)/(precision+recall);
     [~,~,~,AUC_nb(i)] = perfcurve(test_y,score(:,2),2);
     
end

close all;

%% 6



figure;boxplot([AUC_svm, AUC_randfore, AUC_knn, AUC_nb], 'orientation', 'horizontal','Labels',{'SVM','Random-Forest','KNN','Naive Bayes'})
xlabel('Area Under the Curve');
title('AUC for different classifier');


%%assistive function
function [kernel_scale_best,C_best,train_X, train_y, tune_up_X, tune_up_y] = svm_hyperparameter_tuning_new(X,y,kernel,kernel_scales,Cs) 
% performs grid search to find the best combination of sigma (p_best) and
% C (c_best) parameters, trains an SVM model using these parameters and evaluates it on the test data.
accs = zeros(length(kernel_scales), length(Cs));
    
cvp = cvpartition(y, 'Holdout', 0.2); % seperate 25% of the data for validation
training = cvp.training;
tune_up = cvp.test;

train_X = X(training,:); % form the training data
train_y = y(training); 
    
tune_up_X = X(tune_up,:); % form the testing data
tune_up_y = y(tune_up);   

% for each combination of parameters, train a classifier and store the accuracy of classificaiton on the validation data.
for i = 1:length(kernel_scales)
    for j = 1:length(Cs)
        model = fitcsvm(train_X, train_y, 'KernelFunction', kernel, 'KernelScale', kernel_scales(i), 'BoxConstraint', Cs(j));
        test_pred_y = predict(model, tune_up_X); % use the trained model to classify the testing data 
        test_acc = sum(tune_up_y == test_pred_y)/length(tune_up_y); % find the classifiction accuracy on the training data
        accs(i,j) = test_acc;
    end
end
    

[kernel_scale_best_idx,C_best_idx] = find(accs == max(accs(:)));  %find the combination of parameters that has be best accuracy
kernel_scale_best_idx = kernel_scale_best_idx(1); % in case there are more than one combination with max performance
C_best_idx = C_best_idx(1); 

kernel_scale_best = kernel_scales(kernel_scale_best_idx);
C_best = Cs(C_best_idx);
end