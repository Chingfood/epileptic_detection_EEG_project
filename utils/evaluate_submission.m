%% Load the training data
load('EEG_signals.mat')
load('labels.mat')

%% Generate a model based on the training data that you have access to
training_features = generate_features(EEG_signals);
model = generate_model(training_features, labels);

%% Load the unseen test data. You don't have access to this data
load('test_EEG_signals.mat') 
load('test_labels.mat') 

%% Test the model on the test data and return the F1 score
testing_features = generate_features(test_EEG_signals);
results = predict(model, testing_features);

true_positive = sum(test_labels==1 & results==1);
labeled_positive = sum(test_labels==1);
predicted_positive = sum(results==1);

test_F1_score = 2 * true_positive /(labeled_positive + predicted_positive)