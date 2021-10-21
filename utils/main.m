load('EEG_signals.mat');
load('labels.mat');
Fs = 173.61;

s = EEG_signals(1,:);
s_e = EEG_signals(81,:);
d1_mat = zeros(size(EEG_signals));
d2_mat = zeros(size(EEG_signals));
d12_mat = zeros(size(EEG_signals));
for i = 1:160
    [C,L] = wavedec(EEG_signals(i,:),3,'db4');
    C(1 : L(1)+L(2)) = 0; % set a3 d3 to 0
    C_d1 = C;
    C_d1(L(1)+L(2)+1:L(1)+L(2)+L(3)) = 0; %set d2 to 0
    C_d2 = C;
    C_d2(L(1)+L(2)+L(3)+1:L(1)+L(2)+L(3)+L(4)) = 0; %set d1 to 0
    d12_mat(i,:) = waverec(C,L,'db4');
    d1_mat(i,:) = waverec(C_d1,L,'db4');
    d2_mat(i,:) = waverec(C_d2,L,'db4');
end

k1 = zeros(1,160);
k2 = zeros(1,160);
k12 = zeros(1,160);
for i = 1:160
k1(i)=ApEn(2,0.15*std(d1_mat(i,:)),d1_mat(i,:));
k2(i)=ApEn(2,0.15*std(d2_mat(i,:)),d2_mat(i,:));
k12(i)=ApEn(2,0.15*std(d12_mat(i,:)),d12_mat(i,:));
end



figure;gscatter(1:160,k1,labels)

d1 = zeros(160,2052);
d2 = zeros(160,1029);
d12 = zeros(160,3081);
for i = 1:160
    [C,L] = wavedec(EEG_signals(i,:),3,'db4');
    d12(i,:) = C(L(1)+L(2)+1:L(1)+L(2)+L(3)+L(4));
    d1(i,:) =  C(L(1)+L(2)+L(3)+1:L(1)+L(2)+L(3)+L(4));
    d2(i,:) =  C(L(1)+L(2)+1:L(1)+L(2)+L(3));
end
h1 = zeros(1,160);
h2 = zeros(1,160);
h12 = zeros(1,160);
for i = 1:160
h1(i)=ApEn(2,0.15*std(d1(i,:)),d1(i,:));
h2(i)=ApEn(2,0.15*std(d2(i,:)),d2(i,:));
h12(i)=ApEn(2,0.15*std(d12(i,:)),d12(i,:));
end
figure;gscatter(1:160,h1,labels)

pe = zeros(1,160);
for i = 1:160
    pe(i)=pe_3(EEG_signals(i,:),3,1);
end
figure;gscatter(1:160,pe,labels)
pd1 = zeros(1,160);
pd2 = zeros(1,160);
pd12 = zeros(1,160);
pd1_mat = zeros(1,160);
pd2_mat = zeros(1,160);
pd12_mat = zeros(1,160);
for i = 1:160
    pd1(i)=pe_3(d1(i,:),3,1);
    pd1_mat(i) = pe_3(d1_mat(i,:),3,1);
    pd2(i) = pe_3(d2(i,:),3,1);
    pd2_mat(i) = pe_3(d2_mat(i,:),3,1);
    pd12(i) = pe_3(d12(i,:),3,1);
    pd12_mat(i) = pe_3(d12_mat(i,:),3,1);
end

figure; gscatter(1:160,pd1,new_labels);
scatter3(h1,h2,pe,1,new_labels)

features = vertcat(h1,h2,pe);
features = features';

Files=dir('*.txt');
formatSpec = '%d';
EEG_additional = zeros(100,4097);
for i = 1:length(Files)
    fileID = fopen(Files(i).name,'r');
    EEG_additional(i,:) = fscanf(fileID,formatSpec);
    fclose(fileID);
end

EEG_full = vertcat(EEG_additional,EEG_full);
new_labels = zeros(460,1);
new_labels(281:end) = 1;

d1_full = zeros(460,2052);
h1_full = zeros(1,460);
pe_full = zeros(1,460);



for i = 1:460
    pe_full(i)=pe_3(EEG_full(i,ho:),3,1);
    [C,L] = wavedec(EEG_full(i,:),3,'db4');
    d1_full(i,:) =  C(L(1)+L(2)+L(3)+1:L(1)+L(2)+L(3)+L(4));
    h1_full(i)=ApEn(2,0.15*std(d1_full(i,:)),d1_full(i,:));
end

EEG_tests=zeros(300,4097);
cd SET_A
Files=dir('*.txt');
formatSpec = '%d';
for i = 1:length(Files)
    fileID = fopen(Files(i).name,'r');
    EEG_tests(i,:) = fscanf(fileID,formatSpec);
    fclose(fileID);
end
cd ../SET_B
Files=dir('*.txt');
formatSpec = '%d';
for i = 1:length(Files)
    fileID = fopen(Files(i).name,'r');
    EEG_tests(i+100,:) = fscanf(fileID,formatSpec);
    fclose(fileID);
end
cd ../SET_E
Files=dir('*.txt');
formatSpec = '%d';
for i = 1:length(Files)
    fileID = fopen(Files(i).name,'r');
    EEG_tests(i+200,:) = fscanf(fileID,formatSpec);
    fclose(fileID);
end

test_labels = zeros(300,1);
test_labels(201:end) = 1;
d1_test = zeros(300,2052);
h1_test = zeros(1,300);
pe_test = zeros(1,300);



for i = 1:300
    pe_test(i)=pe_3(EEG_tests(i,:),3,1);
    [C,L] = wavedec(EEG_tests(i,:),3,'db4');
    d1_test(i,:) =  C(L(1)+L(2)+L(3)+1:L(1)+L(2)+L(3)+L(4));
    h1_test(i)=ApEn(2,0.15*std(d1_test(i,:)),d1_test(i,:));
end

d2_test = zeros(300,1029);
h2_test = zeros(1,300);
for i = 1:300
    [C,L] = wavedec(EEG_tests(i,:),3,'db4');
    d2_test(i,:) =  C(L(1)+L(2)+1:L(1)+L(2)+L(3));
    h2_test(i)=ApEn(2,0.15*std(d2_test(i,:)),d2_test(i,:));
end
test_X = horzcat(h1_test',pe_test');
test_y = test_labels;
train_X = horzcat(h1',pe');
train_y = labels;

model = fitcsvm(train_X, labels, 'KernelFunction', 'rbf', 'KernelScale', 1); % 'BoxConstraint' parameter, C, is set to 1 as a default

train_pred_y = predict(model, train_X);
train_acc = sum(train_y == train_pred_y)/length(train_y); 
test_pred_y = predict(model, test_X); 
test_acc = sum(test_y == test_pred_y)/length(test_y);
svm_plot_classes(train_X,train_y,model); 
svm_plot_classes(test_X,test_y,model);

nb_and_knn_plot_classes(train_X,train_y,model); 
train_Xh = horzcat(h1',h2',pe');
test_Xh = horzcat(h1_test',h2_test',pe_test);
coeff_train= pca(train_Xh);
coeff_test = pca(test_Xh);
train_Xh = train_Xh*coeff_train(:,1:2);
test_Xh = test_Xh*coeff_train(:,1:2);

model = fitcsvm(train_Xh, labels, 'KernelFunction', 'rbf', 'KernelScale', 1, 'BoxConstraint', 0.5); % 'BoxConstraint' parameter, C, is set to 1 as a default

test_acc=zeros(1,63);
for i = 36:63
    model = fitcknn(train_Xh, labels,'NumNeighbors',i);
    
    results = predict(model, test_Xh);
    true_positive = sum(test_labels==1 & results==1);
    labeled_positive = sum(test_labels==1);
    predicted_positive = sum(results==1);
    test_F1_score = 2 * true_positive /(labeled_positive + predicted_positive);
    test_acc(i) = test_F1_score;
    nb_and_knn_plot_classes(test_X,test_y,model); 
end
model = fitcknn(train_Xh, labels,'NumNeighbors',50);

nb_and_knn_plot_classes(train_X,train_y,model); 