% A Matlab script to augment the LASA dataset with Riemannian (symmetirc and positive definite (SPD) matrices) motion profiles.
%
% The implemented procedure described in Saveriano et al., Learning Stable Robotic Skills on Riemannian Manifolds, 2021. 
%
% 
% Author: Matteo Saveriano
%
% Copyright (c) 2021 Matteo Saveriano, Dept. of Computer Science and 
% Digital Science Center, University of Innsbruck, 6020, Innsbruck,
% Austria, https://iis.uibk.ac.at
%

clear;
close all;

addpath(genpath('libs'))

modelPath = 'LASA_dataset/';

%% Preprocess demonstrations
% Define a SPD goal
S_goal = 100*eye(2,2);

dt = 0.003; % Set an average sampling time for the LASA dataset
for i=1:30
    % Load demonstrations
    [demos, ~, name]   = load_LASA_models(modelPath, i);
    
    % Stack all demos into a matrix
    demoMat = [];
    rangeUQ = [];
    for demoIt=1:length(demos)
        demoMat = [demoMat; demos{demoIt}.pos];
    end
    
    demoSPD = [];
    idx_ = [1:3; 5:7; 9:11; [13,14,1]];
    for demoSPDIt=1:size(idx_,1)
        % Take 1 + 1/2 demonstrations to obtain a 3D motion, skip the rest
        demoSPD{demoSPDIt}.tsPos = demoMat(idx_(demoSPDIt,:),:);
        
        % Compute TS veloctiy
        demoSPD{demoSPDIt}.tsVel = [diff(demoSPD{demoSPDIt}.tsPos,[],2)./dt zeros(3,1)];
        
        % Store the sampling time
        demoSPD{demoSPDIt}.dt = dt;  
        
        % Compute SPD tajectory
        for tt=1:size(demoSPD{demoSPDIt}.tsPos, 2)
            log_spd = demoSPD{demoSPDIt}.tsPos(:,tt);
            demoSPD{demoSPDIt}.spd(:,:,tt) = expmap(S_goal, vec2symMat(log_spd));
        end
    end
    
    filename = ['R_LASA_SPD/' name '_SPD.mat'];
    save(filename, 'demoSPD')
end
