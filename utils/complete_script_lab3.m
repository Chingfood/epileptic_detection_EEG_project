S = load('f543l.mat');
ecg = S.ecg;   % S is a struct that contains the ecg signal and sampling rate. You can access the elements within a struct using the dot syntax.
Fs = S.Fs;

figure;plot(ecg)    % The ecg signal contains sinus rhythm and ventricular fibrillation 

seg = ecg(10000:12000);
figure;plot(seg)

waveletAnalyzer    % The gui for wavelet toolbox

% Select wavelet display -> try different mother wavelets and click display
% to get the shape and other info for that mother wavelet
%% example 1
% Continuous Wavelet 1-D
% load seg: File -> Import signal from workspace -> select seg
% wavelet = db4
% Scale = 1:1:200
% Analyze
% Right click on coefficients -> New coefficient line

%% example 2
% Wavelet 1-D
% load seg: File -> Import from Workspace -> Import signal -> select seg
% wavelet = db4
% level = 8
% Analyze
% try different display modes
% Notice the y scale
% play with denoise
% denoise --> set d1 threshold to 251, d2 threshold to 1826, and d3
% threshold to 8461 --> clock on De-noise --> view denoised signal and
% residuals

%% example 3
figure;plot(ecg)
% 1) select the first 10s of the signal and assign it to y1
% 2) select another 10-second segment of the ecg signal from 240s-250s and assign it to y2 
% 3) visualize y1 and y2
% 4) calculate the fourier transform of y1 and assign it to Y1
% 5) calculate the fourier transform of y2 and assign it to Y2
% 6) define the frequency vector
% 7) overlay Y1 and Y2 in a plot
% 8) add legends for "sinus rythm" and "ventricular fibrilation"
% 9) zoom in frequencies 0-40 Hz
% 10) analyze their differences
y1 = ecg(1:10*Fs);
y2 = ecg(60001:60000+10*Fs);
Y1 = fft(y1);
Y2 = fft(y2);
f = (0:length(Y1)-1)*Fs/(length(Y1)-1);
figure;plot(f,abs(Y1));
hold all; plot(f,abs(Y2));
xlabel('Frequency (Hz)')
ylabel('|Y|')
xlim([0,40]);
legend('Sinus rythm', 'Ventricular fibrilation')
%% example 4
waveletAnalyzer

% Wavelet 1-D
% load ecg
% wavelet = db4
% level = 8
% Analyze
% Check out the amplitude of the sinus rhythm in d1 to d3 and the ampltude 
% of ventricular fibrillation in d5.

%% example 5
% repeat example 4 but this time decompose the ecg signal at level 5. 
% Use De-noise to separate d1+d2+d3 from d4+d5+a5.   

%% example 6 - part 1, using Command Window
% goal : detecting ventricular fibrilation
% how:
% use wavedecom to decompose the ecg signal using wavelet = db4, level = 5
% keep d1, d2, and d3 by excluding a5, d4, and d5. Hint: you can exclude 
% a5, d4, and d5 by setting them to zero!
% reconstruct the signal from d1 to d3 and assign it to an array 'ecg_high'
% compute the power of the ecg_high using 
% >> [ecg_high_pw, tm] = find_power(ecg_high,Fs); command and plot
% ecg_high_pw versus tm
% Can you find any criteria to classify ventricular fibrilation from sinus
% rythm?

[C,L] = wavedec(ecg,5,'db4'); % C is the coefficient vector, while L is the bookkeeping vector
                              % that contains the number of coefficients by
                              % level, L = [an dn dn-1 ... d1]
C(1 : L(1)) = 0; % set a5 to 0
C(L(1)+1: L(1)+L(2)+L(3)) = 0; % set d4 and d5 to zero
ecg_high = waverec(C,L,'db4'); % reconstruct the signal from d1 to d3
figure;
ax1 = subplot(2,1,1);
plot(ax1,ecg_high)
title('reconstructed signal')
ylim([min(ecg) max(ecg)])
ax2 = subplot(2,1,2);
plot(ax2, ecg)
title('original signal')
linkaxes([ax1 ax2], 'xy')

[ecg_high_pw, tm] = find_power(ecg_high,Fs); % compute the power of the wavelet reconstruction
figure;plot(tm, ecg_high_pw);

%% example 6 - part 2
% Is the criteria generalizable to another ventricular fibrilation case
% load the second ventricular fibrilation signal, ecg2
S = load('f563l.mat');  
ecg2 = S.ecg;
figure;plot(ecg2)
% investigate the end of the signal (from sample 68000) for ventricular fibrilation  

% repeat the same process as example 6 but this time for ecg2
% use wavedecom to decompose the ecg2 signal using wavelet = db4, level = 5
% keep d1, d2, and d3 by excluding a5, d4, and d5. 
% reconstruct the signal from d1 to d3 and assign it to an array 'ecg2_high'
% compute the power of the ecg2_high using 
% >> [ecg2_high_pw, tm] = find_power(ecg2_high,Fs); command and plot
% ecg_high_pw versus tm
% Can you find any criteria to classify ventricular fibrilation from sinus
% rythm?

[C,L] = wavedec(ecg2,5,'db4');
C(1 : L(1)) = 0; % set a5 to 0
C(L(1)+1: L(1)+L(2)+L(3)) = 0; % set d4 and d5 to zero
ecg2_high = waverec(C,L,'db4');
figure;plot(ecg2_high);

[ecg2_high_pw, tm] = find_power(ecg2_high,Fs);
figure;plot(tm, ecg2_high_pw);

%% example 6 - part 3
% overlay ecg_high_pw and ecg2_high_pw in one plot
% does that criteria still work? Why?
figure;plot(tm, ecg_high_pw);   % notice how the two power signals are on different scales
hold all;plot(tm, ecg2_high_pw);
legend('ECG1 High Power', 'ECG2 High Power');


%% example 6 - part 4, 

% use wavedec to decompose ecg signal using wavelet = db4, level = 5
% this time only keep d5.
% reconstruct 'ecg_low' from d5 coefficients 
% compute the power of the ecg_low using the "find_power" function
% calculate the relative Power for ecg signal by deviding the high power 
% to the low power  (ecg_high_pw./ecg_low_pw) and assign it to
% ecg_relative_power. 
% repeat the same thing for ecg2 and compute ecg2_relative_power
% overlay ecg_relative_power and ecg2_relative_power in one plot
% Any new criteria to classify ventricular fibrilation from sinus rythms

[C,L] = wavedec(ecg,5,'db4');
C(1:L(1)) = 0;  % set a5 to zero
C(L(1)+L(2)+1:end) = 0;     % set d1 to d4 to zero
ecg_low = waverec(C,L,'db4');   % reconstruct the signal from d5 only 
figure;plot(ecg_low);

[ecg_low_pw, tm] = find_power(ecg_low,Fs);  % compute the power of the wavelet reconstruction
figure;plot(tm, ecg_low_pw);

figure; plot(tm, ecg_high_pw./ecg_low_pw);  % check out the relative power
figure;plot(ecg)

% repeat the same thing for ecg2

[C,L] = wavedec(ecg2,5,'db4');
C(1:L(1)) = 0;
C(L(1)+L(2)+1:end) = 0;
ecg2_low = waverec(C,L,'db4');
figure;plot(ecg2_low);

[ecg2_low_pw, tm] = find_power(ecg2_low,Fs);
figure;plot(tm, ecg2_low_pw);

figure; plot(tm, ecg_high_pw./ecg_low_pw);
hold all; plot(tm, ecg2_high_pw./ecg2_low_pw);
legend('ECG1 relative Power', 'ECG2 relative Power');

close all
clear all

%% Wavelet 2D
%% example 7
image  = imread('building.jpg');
figure; imshow(image);

waveletAnalyzer

% Load image
% wavelet: Haar level 4
% Choose any set of coefficients (box) and then click on reconstruct
% Compare vertical, horizontal and diagonal coefficients
% Compare vertical components at levels 1 and 4
