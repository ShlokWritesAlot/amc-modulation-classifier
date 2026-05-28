clear; clc; 

% BLOCK 1: Parameters

sps = 8;      %samples per symbol
spf = 1024;   %samples per frame (1024/8=128 frames), 10^2= highly efficient for CNN
fs = 200e3;
SNR = 30;
numFramesPerModType = 1000;


% BLOCK 2: Modulation Types.

modTypes = categorical(["BPSK","QPSK","8PSK",...
                        "16QAM", "64QAM", "PAM4",...
                         "GFSK", "CPFSK", "B-FM",...
                         "DSB-AM","SSB-AM"]);

bitsPerSymbol = [1, 2, 3, 4, 6, 2, 2, 2, 2, 2, 2]; % Bits required for each signal 