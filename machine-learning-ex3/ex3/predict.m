function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% ------------My commens:

# inputs are pixel values of digit images. Images are 20x20 - thus we have 400 input layer units
# + extra bias unit which always outputs +1

# the parameters (Thera1, Theta2) have dimensions that are sized for a neural network
# with 25 units in the second layer and 10 output units (10 cases)

% (1) comute h(xi) for every example i and returns associated preds
% (2) pred will be the max one like in the oneVsAll

% OneVsAll predictions:

##size(all_theta)
##z = X * all_theta';
##h = sigmoid(z);
##
##size(h);
##
##for i= 1:m
##  [x, ix] = max(h(i,:));
##  p(i,:) = ix;
##endfor

X_ = ones(size(X,1), size(X,2) + 1);
X_(:,2:end) = X;

z2 = Theta1 * X_';
a2 = sigmoid(z2);

a2_ = ones(size(a2,1)+1, size(a2,2));
a2_(2:end,:) = a2;

z3 = Theta2 * a2_;
h = sigmoid(z3);

size(h)
# did 10 x 5000 instead of 5000 x 10...
for i = 1:m
  [x,ix] = max(h(:,i));
  p(i,:) = ix;
endfor
% =========================================================================


end
