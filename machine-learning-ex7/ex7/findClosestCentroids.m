function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%
m = size(X,1);
% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
dist = zeros(m, K);

for i=1:m
  dst = sum((X(i,:)-centroids(1,:)).^2);

  for j=2:K
    d = sum((X(i,:)-centroids(j,:)).^2);
    dst = [dst d];
  endfor
  #dst
  [dist(i), idx(i)] = min(dst);
endfor




% =============================================================

end

