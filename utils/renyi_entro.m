function y=renyi_entro(DATA,alpha)

%% Calculating probability of each possible value. 
DATA = round(DATA,2); 
unique_DATA = unique(DATA);
P = zeros(length(unique_DATA),1);

for i = 1: length(unique_DATA)
    P(i) = length(find(DATA == unique_DATA(i)))/length(DATA);
end
%% Calculating Renyi Entopy
y=log(sum(P(:).^alpha))/(1-alpha);
