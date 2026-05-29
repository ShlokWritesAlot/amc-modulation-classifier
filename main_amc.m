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


%% Block 4: Frame Generator Loop
numModTypes = length(modTypes);
totalFrames = numModTypes * numFramesPerModType;

% Preallocate dataset
frameData  = zeros(2, spf, 1, totalFrames);
frameLabel = repmat(modTypes(1), totalFrames, 1);
frameIdx = 1;

for modIdx = 1:numModTypes
    modType = modTypes(modIdx);
    bps     = bitsPerSymbol(modIdx);

    % Get the modulator function for this mod type
    modulator = helperModClassGetModulator(modType, sps, fs);

    % How many bits to generate per batch
    numSymbols = 10 * spf;
    numBits    = numSymbols * bps;
    frameCount = 0;

    while frameCount < numFramesPerModType

        % Step 1: Generate random bits and modulate
        bits     = randi([0 1], numBits, 1);
        txSignal = modulator(bits);

        % Step 2: Pass through Rician channel
        rxSignal = ricianChannel(txSignal);

        % Step 3: Apply random clock offset
        release(clockOffset);
        clockOffset.FrequencyOffset = fs * 5e-6 * (2*rand()-1);
        rxSignal = clockOffset(rxSignal);

        % Step 4: Apply AWGN
        release(awgnChannel);
        rxSignal = awgnChannel(rxSignal);

        % Step 5: Slice into frames
        frames = helperModClassFrameGenerator(rxSignal, spf, spf, 50, sps);

        % Step 6: Store each frame
        for k = 1:size(frames, 2)
            if frameCount >= numFramesPerModType
                break;
            end
            frame = frames(:, k);
            frameData(1, :, 1, frameIdx) = real(frame);
            frameData(2, :, 1, frameIdx) = imag(frame);
            frameLabel(frameIdx)         = modType;
            frameIdx   = frameIdx   + 1;
            frameCount = frameCount + 1;
        end

    end

    fprintf('Generated %d frames for %s\n', numFramesPerModType, char(modType));
end

fprintf('Dataset complete: %d total frames\n', totalFrames);


%% Block 5: Train/Test Split

rng(42); %set random seed (random.seed() in python)

partition  = cvpartition(frameLabel, 'HoldOut', 0.2);


% Training Set

trainData = frameData(:, :, :, training(partition));
trainLabel = frameLabel(training(partition));


% Test set

testData  = frameData(:, :, :, test(partition));
testLabel = frameLabel(test(partition));

fprintf('Training frames: %d\n', sum(training(partition)));
fprintf('Test frames:     %d\n', sum(test(partition)));

%% Block 6: Build CNN Architecture 

modCLassNet = helperModClassCNN(modTypes, sps, spf);

% Visualize the network architecture 

analyzeNetwork(modCLassNet)


%% Block 7: Training Options
trainOptions = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

%% Block 8: Train Network
% Reconstruct complex training data from I and Q rows
trainDataComplex = squeeze(trainData(1,:,:,:)) + ...
    1j * squeeze(trainData(2,:,:,:));

% Reshape to [1024 × 1 × 8800] — one sample per page
trainDataComplex = reshape(trainDataComplex, spf, 1, []);

% Train the network
trainedNet = trainnet(trainDataComplex, trainLabel, modCLassNet, ...
    'crossentropy', trainOptions);
save('amc_workspace.mat')
%% Block 9: Evaluate on Test Set
% Reconstruct complex test data
testDataComplex = squeeze(testData(1,:,:,:)) + ...
    1j * squeeze(testData(2,:,:,:));

% Reshape to [1024 × 1 × 2200]
testDataComplex = reshape(testDataComplex, spf, 1, []);

% Get prediction scores from trained network
scores = predict(trainedNet, testDataComplex);

% Get correct class order (alphabetical, matches network output)
networkClasses = categories(trainLabel);

% Convert scores to class labels using correct order
[~, idx] = max(scores, [], 2);
predictions = categorical(networkClasses(idx));

% Calculate overall accuracy
accuracy = mean(predictions == testLabel);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Plot confusion matrix
figure;
confusionchart(testLabel, predictions, ...
    'Title', 'Confusion Matrix — Baseline CNN (SNR = 30dB)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

save('amc_workspace.mat')
%% Block 10: SNR Sweep
snrValues = -10:5:30;
accuracyVsSNR = zeros(1, length(snrValues));
numTestFrames = 200;

for snrIdx = 1:length(snrValues)
    currentSNR = snrValues(snrIdx);
    
    % Generate fresh test frames at this SNR
    sweepData  = zeros(2, spf, 1, numModTypes * numTestFrames);
    sweepLabel = repmat(modTypes(1), numModTypes * numTestFrames, 1);
    
    frameIdx = 1;
    for modIdx = 1:numModTypes
        modType   = modTypes(modIdx);
        bps       = bitsPerSymbol(modIdx);
        modulator = helperModClassGetModulator(modType, sps, fs);
        
        reset(ricianChannel);
        
        frameCount = 0;
        numSymbols = 10 * spf;
        numBits    = numSymbols * bps;
        
        while frameCount < numTestFrames
            bits     = randi([0 1], numBits, 1);
            txSignal = modulator(bits);
            
            % Rician channel
            rxSignal = ricianChannel(txSignal);
            
            % Clock offset
            release(clockOffset);
            clockOffset.FrequencyOffset = fs * 5e-6 * (2*rand()-1);
            rxSignal = clockOffset(rxSignal);
            
            % AWGN at current SNR
            release(awgnChannel);
            awgnChannel.SNR = currentSNR;
            rxSignal = awgnChannel(rxSignal);
            
            % Slice into frames
            frames = helperModClassFrameGenerator(rxSignal, spf, spf, 50, sps);
            
            for k = 1:size(frames, 2)
                if frameCount >= numTestFrames
                    break;
                end
                frame = frames(:, k);
                sweepData(1, :, 1, frameIdx) = real(frame);
                sweepData(2, :, 1, frameIdx) = imag(frame);
                sweepLabel(frameIdx)         = modType;
                frameIdx   = frameIdx   + 1;
                frameCount = frameCount + 1;
            end
        end
    end
    
    % Reconstruct complex data
    sweepDataComplex = squeeze(sweepData(1,:,:,:)) + ...
                       1j * squeeze(sweepData(2,:,:,:));
    sweepDataComplex = reshape(sweepDataComplex, spf, 1, []);
    
    % Get predictions
    scores = predict(trainedNet, sweepDataComplex);
    networkClasses = categories(trainLabel);
    [~, idx] = max(scores, [], 2);
    sweepPredictions = categorical(networkClasses(idx));
    
    % Record accuracy
    accuracyVsSNR(snrIdx) = mean(sweepPredictions == sweepLabel);
    fprintf('SNR = %3d dB → Accuracy = %.2f%%\n', currentSNR, accuracyVsSNR(snrIdx)*100);
end

% Plot accuracy vs SNR
figure;
plot(snrValues, accuracyVsSNR * 100, '-o', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('SNR (dB)');
ylabel('Accuracy (%)');
title('Classification Accuracy vs SNR — Baseline CNN');
grid on;
ylim([0 100]);
xticks(snrValues);

save('amc_workspace.mat')
%% Block 11: Retrain with IQ Imbalance
% Generate new training data with IQ imbalance added to channel
fprintf('Generating frames with IQ imbalance...\n');

iqFrameData  = zeros(2, spf, 1, totalFrames);
iqFrameLabel = repmat(modTypes(1), totalFrames, 1);

frameIdx = 1;

for modIdx = 1:numModTypes
    modType   = modTypes(modIdx);
    bps       = bitsPerSymbol(modIdx);
    modulator = helperModClassGetModulator(modType, sps, fs);
    
    reset(ricianChannel);
    
    frameCount = 0;
    numSymbols = 10 * spf;
    numBits    = numSymbols * bps;
    
    while frameCount < numFramesPerModType
        bits     = randi([0 1], numBits, 1);
        txSignal = modulator(bits);
        
        % Rician channel
        rxSignal = ricianChannel(txSignal);
        
        % Clock offset
        release(clockOffset);
        clockOffset.FrequencyOffset = fs * 5e-6 * (2*rand()-1);
        rxSignal = clockOffset(rxSignal);
        
        % AWGN
        release(awgnChannel);
        awgnChannel.SNR = SNR;
        rxSignal = awgnChannel(rxSignal);
        
        % IQ Imbalance — applied after channel
        ampImbalance   = 1 + 0.05 * randn();           % +-5% gain error
        phaseImbalance = 2 * pi/180 * randn();          % +-2 degrees
        rxI = real(rxSignal);
        rxQ = imag(rxSignal);
        rxSignal = (rxI * ampImbalance) + ...
                   1j*(rxQ*cos(phaseImbalance) + rxI*sin(phaseImbalance));
        
        % Slice into frames
        frames = helperModClassFrameGenerator(rxSignal, spf, spf, 50, sps);
        
        for k = 1:size(frames, 2)
            if frameCount >= numFramesPerModType
                break;
            end
            frame = frames(:, k);
            iqFrameData(1, :, 1, frameIdx) = real(frame);
            iqFrameData(2, :, 1, frameIdx) = imag(frame);
            iqFrameLabel(frameIdx)         = modType;
            frameIdx   = frameIdx   + 1;
            frameCount = frameCount + 1;
        end
    end
    fprintf('IQ: Generated %d frames for %s\n', numFramesPerModType, char(modType));
end

% Split into train/test
iqPartition = cvpartition(iqFrameLabel, 'HoldOut', 0.2);

iqTrainData  = iqFrameData(:,:,:,training(iqPartition));
iqTrainLabel = iqFrameLabel(training(iqPartition));
iqTestData   = iqFrameData(:,:,:,test(iqPartition));
iqTestLabel  = iqFrameLabel(test(iqPartition));

% Reconstruct complex
iqTrainComplex = squeeze(iqTrainData(1,:,:,:)) + ...
                 1j * squeeze(iqTrainData(2,:,:,:));
iqTrainComplex = reshape(iqTrainComplex, spf, 1, []);

iqTestComplex  = squeeze(iqTestData(1,:,:,:)) + ...
                 1j * squeeze(iqTestData(2,:,:,:));
iqTestComplex  = reshape(iqTestComplex, spf, 1, []);

% Retrain network with IQ impaired data
iqNet = trainnet(iqTrainComplex, iqTrainLabel, modCLassNet, ...
                 'crossentropy', trainOptions);

% Evaluate
iqScores = predict(iqNet, iqTestComplex);
networkClasses = categories(iqTrainLabel);
[~, iqIdx] = max(iqScores, [], 2);
iqPredictions = categorical(networkClasses(iqIdx));

iqAccuracy = mean(iqPredictions == iqTestLabel);
fprintf('IQ Imbalance Test Accuracy: %.2f%%\n', iqAccuracy * 100);

% Confusion matrix
figure;
confusionchart(iqTestLabel, iqPredictions, ...
    'Title', 'Confusion Matrix — CNN with IQ Imbalance (SNR = 30dB)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
save('amc_workspace.mat')
%% Block 12: Extended Modulation Types
fprintf('Generating extended modulation types...\n');

% Extended modulation list — original 11 + 4 new
extModTypes = categorical(["BPSK", "QPSK", "8PSK", ...
                            "16QAM", "64QAM", "PAM4", ...
                            "GFSK", "CPFSK", ...
                            "B-FM", "DSB-AM", "SSB-AM", ...
                            "PI2BPSK", "256QAM", "LTEOFDM", "WLANOFDM"]);

extBitsPerSymbol = [1, 2, 3, 4, 6, 2, 2, 2, 2, 2, 2, 1, 8, 4, 4];

numExtModTypes   = length(extModTypes);
numExtFrames     = numFramesPerModType;
totalExtFrames   = numExtModTypes * numExtFrames;

extFrameData  = zeros(2, spf, 1, totalExtFrames);
extFrameLabel = repmat(extModTypes(1), totalExtFrames, 1);

frameIdx = 1;

for modIdx = 1:numExtModTypes
    modType = extModTypes(modIdx);
    bps     = extBitsPerSymbol(modIdx);
    
    frameCount = 0;
    
    while frameCount < numExtFrames
        
        % Generate signal based on modulation type
        if modType == "PI2BPSK"
            numSymbols = 10 * spf;
            bits       = randi([0 1], numSymbols * 1, 1);
            symbols    = pskmod(bits, 2);
            % Apply pi/2 rotation to each symbol
            rotations  = exp(1j * pi/2 * (0:length(symbols)-1)');
            symbols    = symbols .* rotations;
            filterCoeffs = rcosdesign(0.35, 4, sps);
            txSignal   = filter(filterCoeffs, 1, upsample(symbols, sps));
            
        elseif modType == "256QAM"
            % 256-QAM using standard MATLAB
            numSymbols = 10 * spf;
            bits       = randi([0 1], numSymbols * bps, 1);
            symbols    = qammod(bi2de(reshape(bits, bps, []).'), 256, ...
                               'UnitAveragePower', true);
            filterCoeffs = rcosdesign(0.35, 4, sps);
            txSignal   = filter(filterCoeffs, 1, upsample(symbols, sps));
            
        elseif modType == "LTEOFDM"
            % LTE OFDM subframe using LTE Toolbox
            cfg.NDLRB        = 6;
            cfg.CyclicPrefix = 'Normal';
            cfg.DuplexMode   = 'FDD';
            cfg.NCellID      = 1;
            cfg.NSubframe    = 0;
            cfg.NFrame       = 0;
            cfg.CellRefP     = 1;        % number of cell reference ports
            resourceGrid     = lteDLResourceGrid(cfg);
            resourceGrid(:)  = qammod(randi([0 3], numel(resourceGrid), 1), 4, ...
                                     'UnitAveragePower', true);
            [txWaveform, ~]  = lteOFDMModulate(cfg, resourceGrid);
            txSignal         = txWaveform;
            
        elseif modType == "WLANOFDM"
            % 802.11 OFDM using WLAN Toolbox
            cfg         = wlanNonHTConfig('MCS', 3);
            txBits      = randi([0 1], 1000, 1);
            txSignal    = wlanWaveformGenerator(txBits, cfg);
            
        else
            % Original 11 modulation types — use existing helper
            modulator  = helperModClassGetModulator(modType, sps, fs);
            numSymbols = 10 * spf;
            numBits    = numSymbols * bps;
            bits       = randi([0 1], numBits, 1);
            txSignal   = modulator(bits);
        end
        
        % Make sure signal is long enough
        if length(txSignal) < spf + 50
            continue;
        end
        
        % Pass through channel
        sigLen   = length(txSignal);
        rxSignal = txSignal;
        
        % Rician — only apply if signal length matches
        try
            reset(ricianChannel);
            rxSignal = ricianChannel(rxSignal);
        catch
            rxSignal = txSignal;
        end
        
        % Clock offset
        release(clockOffset);
        clockOffset.FrequencyOffset = fs * 5e-6 * (2*rand()-1);
        rxSignal = clockOffset(rxSignal);
        
        % AWGN
        release(awgnChannel);
        awgnChannel.SNR = SNR;
        rxSignal = awgnChannel(rxSignal);
        
        % Slice into frames
        frames = helperModClassFrameGenerator(rxSignal, spf, spf, 50, sps);
        
        for k = 1:size(frames, 2)
            if frameCount >= numExtFrames
                break;
            end
            frame = frames(:, k);
            extFrameData(1, :, 1, frameIdx) = real(frame);
            extFrameData(2, :, 1, frameIdx) = imag(frame);
            extFrameLabel(frameIdx)         = modType;
            frameIdx   = frameIdx   + 1;
            frameCount = frameCount + 1;
        end
    end
    fprintf('Extended: Generated %d frames for %s\n', numExtFrames, char(modType));
end

% Split into train/test
extPartition = cvpartition(extFrameLabel, 'HoldOut', 0.2);

extTrainData  = extFrameData(:,:,:,training(extPartition));
extTrainLabel = extFrameLabel(training(extPartition));
extTestData   = extFrameData(:,:,:,test(extPartition));
extTestLabel  = extFrameLabel(test(extPartition));

% Build new CNN for 15 classes
extNet = helperModClassCNN(extModTypes, sps, spf);

% Reconstruct complex
extTrainComplex = squeeze(extTrainData(1,:,:,:)) + ...
                  1j * squeeze(extTrainData(2,:,:,:));
extTrainComplex = reshape(extTrainComplex, spf, 1, []);

extTestComplex  = squeeze(extTestData(1,:,:,:)) + ...
                  1j * squeeze(extTestData(2,:,:,:));
extTestComplex  = reshape(extTestComplex, spf, 1, []);

% Train
extTrainedNet = trainnet(extTrainComplex, extTrainLabel, extNet, ...
                         'crossentropy', trainOptions);

% Evaluate
extScores = predict(extTrainedNet, extTestComplex);
extClasses = categories(extTrainLabel);
[~, extIdx] = max(extScores, [], 2);
extPredictions = categorical(extClasses(extIdx));

extAccuracy = mean(extPredictions == extTestLabel);
fprintf('Extended Mod Types Accuracy: %.2f%%\n', extAccuracy * 100);

% Confusion matrix
figure;
confusionchart(extTestLabel, extPredictions, ...
    'Title', 'Confusion Matrix — Extended Mod Types (SNR = 30dB)', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

save('amc_workspace.mat')
%% Block 13: Final Comparison Plots

% --- Plot 1: SNR Sweep Curve (already plotted in Block 10, replot cleanly) ---
figure;
plot(snrValues, accuracyVsSNR * 100, '-o', ...
    'LineWidth', 2, 'MarkerSize', 8, 'Color', [0 0.447 0.741]);
xlabel('SNR (dB)');
ylabel('Accuracy (%)');
title('Baseline CNN — Classification Accuracy vs SNR');
grid on;
ylim([0 100]);
xticks(snrValues);
yticks(0:10:100);

% --- Plot 2: Accuracy Comparison Bar Chart ---
figure;
accuracies = [accuracy*100, iqAccuracy*100, extAccuracy*100];
labels     = {'Baseline CNN', 'CNN + IQ Imbalance', 'Extended Mod Types'};
bar(accuracies, 0.5, 'FaceColor', [0 0.447 0.741]);
set(gca, 'XTickLabel', labels);
ylabel('Accuracy (%)');
title('Accuracy Comparison Across Experiments (SNR = 30dB)');
ylim([0 100]);
grid on;

% Add value labels on top of each bar
for i = 1:length(accuracies)
    text(i, accuracies(i) + 1, sprintf('%.2f%%', accuracies(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% --- Print Summary Table ---
fprintf('========== FINAL RESULTS SUMMARY ==========\n');
fprintf('Experiment                  | Accuracy\n');
fprintf('----------------------------+---------\n');
fprintf('Baseline CNN (SNR=30dB)     | %.2f%%', accuracy*100);
fprintf('CNN + IQ Imbalance          | %.2f%%\n', iqAccuracy*100);
fprintf('Extended Mod Types (15)     | %.2f%%\n', extAccuracy*100);
fprintf('============================================\n');
fprintf('SNR Sweep Results:');
for i = 1:length(snrValues)
    fprintf('\nSNR = %3d dB → %.2f%%', snrValues(i), accuracyVsSNR(i)*100);
end
