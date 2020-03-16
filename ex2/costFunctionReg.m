function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%Regularized cost function
denominator = exp(1).^(-1 * (X * theta));
hypot = 1 ./ (1+denominator);

beforeSum = -y .* log(hypot) - ((1-y) .* log(1 - hypot));
J = ((1 / m) * (sum(beforeSum))) + ((lambda / (2 * m)) * sum(theta(2:end) .^ 2));


%Regularized gradient function
beforeSumOne = (hypot - y) .* X(:,1);
beforeSumTwo = (hypot - y) .* X(:,2:end);
firstGrad = (1 / m) * sum(beforeSumOne);
otherGrads = ((1 / m) * sum(beforeSumTwo,1)) + (((lambda / m) * theta(2:end))');
grad = [firstGrad otherGrads];
% =============================================================

end
