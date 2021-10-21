function lab2_fft_filtering


% Fourier transform
%% 

Fs = 100; % sampling rate
L = 50; % length of the signal in seconds
T = 1/Fs; % inter-sample time
t = 0:T:L;
t = t(1:end-1); % exclude the last element of t since time starts at 0 and ends one sample before L seconds
y = 2*sin(2*pi*2*t) + 0.3*sin(2*pi*20*t);
figure;plot(t(1:5*Fs),y(1:5*Fs))

Y = fft(y); % fast fourier transform
magnitude_Y = abs(Y);  % you should use abs(Y) if you want to plot the magnitude of the fourier transform
figure;plot(magnitude_Y)

f = (0:length(Y)-1)*Fs/(length(Y)); % form the vector of frequencies such that size(f) = size(Y) and the first and last elements in f are 0 and Fs, respectively.
figure;plot(f,magnitude_Y)
xlabel('Frequency (Hz)');
ylabel('|Y|')
title('Fourier transform of y(t)')
clc

% fftshift function performs a zero-centered, circular shift on the transform
fshift = (-length(Y)/2:length(Y)/2-1)*Fs/(length(Y)); % zero-centered frequency range
yshift = fftshift(Y);
plot(fshift,abs(yshift))

%% filtering
[bl,al] = butter(3,10*2/Fs); % Butterworth filter (low-pass by defualt). The first input is the filter oreder, 
                             % the 2nd one is the cutoff frequency
y_low = filter(bl,al,y); % Apply filter
figure;plot(y_low(1:5*Fs))

[bl,al] = butter(3,10*2/Fs,'high'); % high-pass filter. filtertype can be:'low', 'bandpass', 'high' or 'stop'
y_high = filter(bl,al,y);
figure;plot(y_high(1:5*Fs))
ylim([-2.5 2.5])
ylim([-0.4 0.3])

[bl,al] = butter(5,10*2/Fs,'high');
y_high = filter(bl,al,y);
figure;plot(y_high(1:5*Fs))

[bl,al] = butter(3,[2 5]*2/Fs,'bandpass'); % band-pass filter between 2 and 5Hz
y_band = filter(bl,al,y);
figure;plot(y_band(1:5*Fs))
close all
clear
clc

%% loading and filtering signals 
load('ecg.mat')

figure;plot(ecg); % the signal contains drift, respiratory component and power-line noise

y = ecg(1:10*Fs);
t = (0:length(y)-1)/Fs;
figure;plot(t,y);

Y = fft(y);
magnitude_Y = abs(Y);
f = (0:length(Y)-1)*Fs/(length(Y));
figure;plot(f,magnitude_Y)

[bl,al] = butter(5,[0.7 40]*2/Fs,'bandpass'); % filtering between 0.7 and 40Hz would remove most of the unwanted components
figure;plot(t,filter(bl,al,y)) % a filter with order=5 is unstable

[bl,al] = butter(3,[0.7 40]*2/Fs,'bandpass');
y2 = filter(bl,al,y);
figure;plot(t,y2,'r')
hold on;plot(t,y)

[bl,al] = butter(3,[55 65]*2/Fs,'stop');
y3 = filter(bl,al,y2);
plot(t,y3,'g')

[hr, t] = find_hr2(ecg,Fs);
figure;plot(t,hr, 'o')
ylim([50 80])

[hr, t] = find_hr2(ecg,Fs,50,120);
figure;plot(t,hr)
ylim([50 80])

close all


