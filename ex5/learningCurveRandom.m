function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)
%Optional exercise 3.5, generalizes the test set and CV set error

% Number of training examples
m = size(X, 1);
r = size(Xval,1);  % the number of validation examples

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

error_train_curr = zeros(m, 1);
error_val_curr   = zeros(r, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

%Hypothesis calculations



%Filling in error vectors
for i = 1:m
    
    for j = 1:50
        X = randperm(m,i);
        Xval = randperm(r,i);
        theta = trainLinearReg(X(1:i,:),y(1:i),lambda);
        error_train_curr(j) = linearRegCostFunction(X(1:i,:),y(1:i),theta,0);
        error_val_curr(j) = linearRegCostFunction(Xval,yval,theta,0);
    end
    
    error_train(i) = mean(error_train_curr);
    error_val_curr(j) = mean(error_val_curr);
    
end





% -------------------------------------------------------------

% =========================================================================

end
