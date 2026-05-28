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
