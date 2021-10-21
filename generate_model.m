function model = generate_model(features,labels)
    
    model = fitcknn(features, labels,'NumNeighbors',50);
end