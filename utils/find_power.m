function [pw, tm] = find_power( sig, Fs )

    win_size = 4*Fs; 
    
    pw = []; 
    tm = [];
    
    for i = 1:length(sig)-win_size+1
        tm(end+1) = i + round(win_size/2);
        pw(end+1) = mean(sig(i:i+win_size-1).^2); 
    end

end

