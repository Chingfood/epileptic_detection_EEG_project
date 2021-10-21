d1 = zeros(1,2052);
d2 = zeros(1,1029);
d12 = zeros(1,3081);
for i = 1:160
    [C,L] = wavedec(EEG_signals(i,:),3,'db4');
    d12(i,:) = C(L(1)+L(2)+1:L(1)+L(2)+L(3)+L(4));
    d1(i,:) =  C(L(1)+L(2)+L(3)+1:L(1)+L(2)+L(3)+L(4));
    d2(i,:) =  C(L(1)+L(2)+1:L(1)+L(2)+L(3));
end

Y=fft(s_e);
magnitude_Y = abs(Y);  % you should use abs(Y) if you want to plot the magnitude of the fourier transform
f = (0:length(Y)-1)*Fs/(length(Y)); % form the vector of frequencies such that size(f) = size(Y) and the first and last elements in f are 0 and Fs, respectively.
fshift = (-length(Y)/2:length(Y)/2-1)*Fs/(length(Y)); % zero-centered frequency range
yshift = fftshift(Y);
plot(fshift,abs(yshift))
xlabel('Frequency (Hz)');
ylabel('|Y|')
title('Fourier transform of y(t)')