function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
n=sigmoid(X*theta);
a=log(n);
b=log(1-n); 
t2=theta;
t2(1,1)=0;
t3=t2'*t2;
J = J = ((-(y'*a)-((1-y)'*b))/m)+(lambda/(2*m))*t3;
grad = zeros(size(theta));
grad=((X'*(n-y))/m)+(lambda/m)*t2;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
