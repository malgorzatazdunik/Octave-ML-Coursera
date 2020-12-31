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

S = 0;
h = 0;
g = 0;
  

  #S = ( -y * log(h) - (1-y) * mean((log(1-h))) );

  for (i = 1:m)
    h = sigmoid(X, theta, i);
    S += -y(i,:) * log(h) - (1-y(i,:)) * log(1-h);
    g += X(i,:)' * (h-y(i,:))';
  endfor

  
  thetaS = 0;
  for (j = 2:size(theta,1))
    thetaS += theta(j,:).^2 ;
  endfor
  
J = S/m + (lambda*thetaS)/(2*m);

  gr = g/m;
##  grad = [gr(1,:); gr(2,:)+ (lambda*theta(2,:))/m; gr(3,:)+ (lambda*theta(3,:))/m]
  for (k = 2:size(gr,1))
    gr(k,:) += (lambda*theta(k,:))/m;
  endfor
  grad= gr;
  



  function h = sigmoid(X, theta, i)
    z = theta' * X(i,:)';
    h = 1 ./ (1 + exp(-z));
  endfunction




% =============================================================

end
