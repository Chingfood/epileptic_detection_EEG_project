% Lab4

%% Entropy
Fs = 100; % sampling rate
L = 25; % length of the signal in seconds
T = 1/Fs; % inter-sample time
t = 0:T:L;
t = t(1:end-1);
y = 2*sin(2*pi*2*t);
y(20*Fs+1:end) = 2*randn(500, 1);
plot(t, y)
window_size = 2*Fs;
% Renyi entropy
renyiEntropy = [];
t_entropy = [];
alpha = 2;
for i = 1:length(t)-window_size
    renyiEntropy(end+1) = renyi_entro(y(i:i+window_size-1)',alpha);
    t_entropy(end+1) = (i + window_size/2-1)/Fs;
end
figure;plot(t, y)
hold on;plot(t_entropy, renyiEntropy)

clear all
close all

% Exercise
% load f543l.mat and plot the signal
% calculate the renyi entropy with alpha = 2 for the ecg signal by 
% sliding a 15-second window with 1-second steps
load('f543l.mat');
figure;plot(ecg)
alpha = 2;
window_size = 15 * Fs;
step = 1 * Fs;
t_ecg = (1:length(ecg))/Fs;
t_entropy = [];
renyiEntropy = [];

for i = 1: step : length(ecg) - window_size
    renyiEntropy(end+1) = renyi_entro(ecg(i:i+window_size-1),alpha);
    t_entropy(end+1) = (i + window_size/2-1)/Fs;
end

% generate a plot with two panels to show ecg and the entropy of the signal
figure; ax1 = subplot(2,1,1);
plot(ax1, t_ecg, ecg)
ax2 = subplot(2,1,2);
plot(ax2, t_entropy, renyiEntropy)

clear
clc
close all

%% Auto-correlation
Fs = 100; 
L = 25; 
T = 1/Fs; 
t = 0:T:L;
t = t(1:end-1);
y1 = 2*sin(pi*2*t);
[acor, lag] = xcorr(y1, 'unbiased');  %lag is a vector with the lags at 
% which the correlations are computed.
figure;subplot(2,1,1)
plot(t, y1)
subplot(2,1,2)
plot(lag/Fs,acor)

%% cross-correlation
y2 = 2*sin(pi*2*t+0.5*pi);
figure;plot(t,y1)
hold on;plot(t,y2)

[ccor,lag] = xcorr(y1,y2);
[~,I] = max(ccor);
lagDiff = lag(I)
timeDiff = lagDiff/Fs

% using cross-correlation to align
load('17453.mat')

t1 = (0:length(signal1)-1)/Fs;
t2 = (0:length(signal2)-1)/Fs;

figure;subplot(2,1,1)
plot(t1,signal1)
title('signal1')

subplot(2,1,2)
plot(t2,signal2)
title('signal2')
xlabel('Time (s)')

[ccor,lag] = xcorr(signal1,signal2,'unbiased');
[~,I] = max(ccor);
lagDiff = lag(I)
timeDiff = lagDiff/Fs

figure %x-axis is in seconds
plot(lag/Fs,ccor)
g1 = gca;
g1.XTick = sort([-15:5:15 timeDiff]);

signal2_aligned = signal2;
signal2_aligned(1 : abs(lagDiff)) = []; 
t2_aligned = (0:length(signal2_aligned)-1)/Fs;

figure;subplot(2,1,1)
plot(t1,signal1)
title('signal1')
xlim([min(t1) max(t1)])

subplot(2,1,2)
plot(t2_aligned,signal2_aligned)
title('signal2, aligned')
xlabel('Time (s)')
xlim([min(t1) max(t1)])

% change mean of one signal
signal1 = signal1 + 1000;
signal2 = signal2 + 1000;

figure;subplot(2,1,1)
plot(t1,signal1)
title('signal1')

subplot(2,1,2)
plot(t2,signal2)
title('signal2')
xlabel('Time (s)')

[ccor2,lag2] = xcorr(signal1,signal2,'unbiased');
[~,I2] = max(ccor2);
lagDiff2 = lag2(I2)
timeDiff2 = lagDiff2/Fs

figure
plot(lag2,ccor2)
g1 = gca;
g1.XTick = sort([-3000:1000:3000]);

%% cross-covariance
[ccov,lag_cov] = xcov(signal1,signal2,'unbiased');
[~,I_cov] = max(ccov);
lagDiff_cov = lag_cov(I_cov)
timeDiff_cov = lagDiff_cov/Fs

figure;plot(lag_cov,ccov)
g3 = gca;
g3.XTick = sort([-3000:1000:3000 lagDiff_cov]);

signal2_aligned_cov = signal2;
signal2_aligned_cov(1: abs(lagDiff_cov)) =[];
t2_aligned_cov = (0:length(signal2_aligned_cov)-1)/Fs;

figure;subplot(2,1,1)
plot(t1,signal1)
title('signal1')
xlim([min(t1) max(t1)])

subplot(2,1,2)
plot(t2_aligned_cov,signal2_aligned_cov)
title('signal2, aligned')
xlabel('Time (s)')
xlim([min(t1) max(t1)])

close all
clear all 
clc