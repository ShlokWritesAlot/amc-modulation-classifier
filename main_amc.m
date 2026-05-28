clear; clc; 

%% BLOCK 1: Parameters

sps = 8;      %samples per symbol
spf = 1024;   %samples per frame (1024/8=128 frames), 10^2= highly efficient for CNN
fs = 200e3;
SNR = 30;
numFramesPerModType = 1000;


%% BLOCK 2: Modulation Types.

modTypes = categorical(["BPSK","QPSK","8PSK",...
                        "16QAM", "64QAM", "PAM4",...
                         "GFSK", "CPFSK", "B-FM",...
                         "DSB-AM","SSB-AM"]);

bitsPerSymbol = [1, 2, 3, 4, 6, 2, 2, 2, 2, 2, 2]; % Bits required for each signal 


%% BLOCK 3: Channel Setup (3 Impairments)

% AWGN Channel - adding Gaussian noise, dictated by SNR

awgnChannel = comm.AWGNChannel(...
    'NoiseMethod', 'Signal to noise ratio (SNR)', ...
    'SNR', SNR);
    

% Rician multipath channel - simulates signal reflections 

ricianChannel = comm.RicianChannel(...
    "SampleRate", fs, ...
    'PathDelays',[0 1.8e-6 3.4e-6], ...
    'AveragePathGains', [0 -2 -10], ...
    'KFactor', 4, ...
    'MaximumDopplerShift', 4);


% Clock offset - simulating timing mismatch between transmitter & receiver


clockOffset = comm.PhaseFrequencyOffset(...
    'SampleRate', fs, ....
    'FrequencyOffset', 0);


%% BLOCK 4: Frame Generator Loop

numModTypes = length(modTypes);  % = 11 (for baseline version)
totalFrames = numModTypes * numFramesPerModType; % 11,000 frames

% Preallocating dataset arrays (faster than loop)

frameData = zeroes(2, spf, 1, totalFrames);  % [2x1024x1x11000] refer to notes
frameLabel = repmat(modeTypes(1), totalFrames, 1);

frameIdx = 1; %global frame counter 

for modIdx = 1:numModTypes

    modType = modTypes(modIdx); % example: "BPSK"
    bps  = bitsPerSymbol(modIdx);  %bits per symbol for the mod type BPSK = 1

    for frameNum = 1:numFramesPerModType

        % Step 1: Generating random bits here

        numSymbols = spf/sps; % 128 symbols per frame, refer to notes 
        numBits = numSymbols*bps;  %Total bits needed
        bits = randi([0 1], numBits, 1);

        % Step 2: Modulate bits ==> Complex symbols 
        symbols = helperModClassModulate(bits, modType, bps);


        % Step 3: Pulse shape into upsampled to sps samples
        
        txtSignal = helperModClassPulseShape(symbols, sps);

        % Step 4.1: Rician multipath channel

        rxSignal = ricianChannel(txtSignal);

        % Step 4.2: Randomize clock offset per frame (0 earlier) +- ppm

        release(clockOffset);
        clockOffset.FrequencyOffset = fs * 5e-6 * (2*rand()-1);
        rxSignal = clockOffset(rxSignal);


        % Step 4c: AWGN noise
        release(awgnChannel);
        rxSignal = awgnChannel(rxSignal);


        % Step 5: Clice to spf samples 
        rxSignal = rxSignal(1:spf);

        %step 6: Normalize (zero mean, unit variance)

        rxSignal = rxSignal - mean(rxSignal);
        rxSignal = rxSignal /std(rxSignal);

        % Step 7: Store as [2 × 1024] (I and Q rows)
        frameData(1, :, 1, frameIdx) = real(rxSignal);  % I channel
        frameData(2, :, 1, frameIdx) = imag(rxSignal);  % Q channel
        frameLabel(frameIdx)         = modType;

        frameIdx = frameIdx + 1;  % move to next frame slot


    end
    fprintf('Generated %d frames for %s', numFramesPerModType, modType);

end 





