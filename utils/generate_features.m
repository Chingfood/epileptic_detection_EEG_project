function features = generate_features(data)
    len=size(data,1);

    d1 = zeros(len,2052);
    d2 = zeros(len,1029);
    h1 = zeros(1,len);
    h2 = zeros(1,len);
    pe = zeros(1,len);

    for i = 1:len
        pe(i)=pec(data(i,:),3,1);
        [C,L] = wavedec(data(i,:),3,'db4');
        d1(i,:) =  C(L(1)+L(2)+L(3)+1:L(1)+L(2)+L(3)+L(4));
        d2(i,:) =  C(L(1)+L(2)+1:L(1)+L(2)+L(3));
    end

    for i = 1:len
        h1(i)=ApEn(2,0.15*std(d1(i,:)),d1(i,:));    
        h2(i)=ApEn(2,0.15*std(d2(i,:)),d2(i,:));
    end


    train_Xh = horzcat(h1',h2',pe');

    coeff_train= pca(train_Xh);
    features = train_Xh*coeff_train(:,1:2);
end






























function apen = ApEn( dim, r, data, tau )
%ApEn
%   dim : embedded dimension
%   r : tolerance (typically 0.2 * std)
%   data : time-series data
%   tau : delay time for downsampling


%---------------------------------------------------------------------
% This code is refered from Kijoon Lee,  kjlee@ntu.edu.sg to calculate 
% approximate entropy from matlab module.

%---------------------------------------------------------------------
if nargin < 4, tau = 1; end
if tau > 1, data = downsample(data, tau); end
    
N = length(data);
result = zeros(1,2);

for j = 1:2
    m = dim+j-1;
    phi = zeros(1,N-m+1);
    dataMat = zeros(m,N-m+1);
    
    % setting up data matrix
    for i = 1:m
        dataMat(i,:) = data(i:N-m+i);
    end
    
    % counting similar patterns using distance calculation
    for i = 1:N-m+1
        tempMat = abs(dataMat - repmat(dataMat(:,i),1,N-m+1));
        boolMat = any( (tempMat > r),1);
        phi(i) = sum(~boolMat)/(N-m+1);
    end
    
    % summing over the counts
    result(j) = sum(log(phi))/(N-m+1);
end

apen = result(1)-result(2);

end

function [pe hist] = pec(y,m,t)

%  Calculate the permutation entropy

%  Input:   y: time series;
%           m: order of permuation entropy
%           t: delay time of permuation entropy, 

% Output: 
%           pe:    permuation entropy
%           hist:  the histogram for the order distribution

% This code is refered from G Ouyang to calculate permutation entropy from
% Matlab module.

%Ref: G Ouyang, J Li, X Liu, X Li, Dynamic Characteristics of Absence EEG Recordings with Multiscale Permutation %     %                             Entropy Analysis, Epilepsy Research, doi: 10.1016/j.eplepsyres.2012.11.003
%     X Li, G Ouyang, D Richards, Predictability analysis of absence seizures with permutation entropy, Epilepsy %     %                            Research,  Vol. 77pp. 70-74, 2007



ly = length(y);
permlist = perms(1:m);
c(1:length(permlist))=0;
    
 for j=1:ly-t*(m-1)
     [a,iv]=sort(y(j:t:j+t*(m-1)));
     for jj=1:length(permlist)
         if (abs(permlist(jj,:)-iv))==0
             c(jj) = c(jj) + 1 ;
         end
     end
 end

hist = c;
 
c=c(find(c~=0));
p = c/sum(c);
pe = -sum(p .* log(p));
end