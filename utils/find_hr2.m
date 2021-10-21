function [hr, t] = find_hr2(ecg,Fs,min_hr,max_hr)
%     This function finds the heart rate from an ECG signal
%     Outputs
%     hr: the heart rate
%     t: time of the detected heart rates
%     
%     Required inputs:
%     ecg: the ECG signal
%     Fs: sampling rate
%     
%     Optional inputs:
%     min_hr: minimum hr in bpm (default = 0.7*60)
%     max_hr: maximum hr in bpm (default = 150)
%

    if(nargin < 3) % if number of input arguments is less than 3 use the default range for heart rate
        min_hr = 0.7*60; %bpm
        max_hr = 150; %bpm
    end
    
    win_size = 10*Fs; % increase this to increase the resolution of the hr detection
    
    step = 1*Fs; % decrease this to detect more hr values within the same time interval
    
    hr = [];
    t = [];
    
    for i = 1:step:length(ecg)-win_size+1 % i specifies the beginning of the window
        win = ecg(i:i+win_size-1);
        
        t(end+1) = (i+win_size/2)/Fs; % time in the middle of the window
        
        [bl,al] = butter(3,[55 65]*2/Fs,'stop'); % remove the 60Hz powerline noise
        win2 = filter(bl,al,win);
        
        [bl,al] = butter(3,0.7*2/Fs,'high'); % remove the respiratory components and drift
        win3 = filter(bl,al,win2);
        
        Y = fft(win3);
        f = (0:length(Y)-1)*Fs/(length(Y));
        hr_range = find(f > min_hr/60 & f < max_hr/60); % find the indices that correspond to the desired range of hr
        f(hr_range)
        
        pw = conj(Y).*Y; % find the power of the frequency components
         
        [~,ix] = max(pw(hr_range)); % find the index that corresponds to the frequency with maximum amplitude within the desired range
        
        hr(end+1) = f(ix+hr_range(1)-1)*60; % change the reference for ix from the beginning of the range to the beginning of the signal and find the corresponding frequency
        
    end

end

