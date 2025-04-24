% A matlab functions to load samples from the LASA HandWritten dataset
%
% Author: Matteo Saveriano
%
% Copyright (c) 2021 Matteo Saveriano, Dept. of Computer Science and 
% Digital Science Center, University of Innsbruck, 6020, Innsbruck,
% Austria, https://iis.uibk.ac.at
%

function [demos, dt, name] = load_LASA_models(datasetPath, modelNumber)
    names = {'Angle','BendedLine','CShape','GShape', 'heee',...
             'JShape','Khamesh','LShape','NShape','PShape',...
             'RShape','Saeghe','Sharpc','Sine','Snake',...
             'Spoon','Sshape','Trapezoid','Worm','WShape','Zshape',...
             'JShape_2', 'Leaf_1', 'Leaf_2', 'DoubleBendedLine','Line',...
             'Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4'};
         
    if(modelNumber<1 || modelNumber>30)
        disp('Wrong model number!')
        error('Please try again and type a number between 1-30.')
    end
    
    load([datasetPath names{modelNumber}],'demos', 'dt'); % loading the model
    
    name = names{modelNumber};
end
