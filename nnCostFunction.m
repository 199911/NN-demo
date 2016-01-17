function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


[r,c] = size(X);
X = cat(2,ones(r,1),X);
A2 = sigmoid(X * Theta1');

[r,c] = size(A2);
A2 = cat(2,ones(r,1), A2);
A3 = sigmoid(A2 * Theta2');

H = A3;
Y = zeros(size(H));
for i = 1:size(y)
    Y(i,y(i)) = 1;
end

J = sum(sum( (-Y) .* log(H) - (1-Y) .* log(1-H))) / m;

[r,c] = size(Theta1);
TmpTheta1 = Theta1(1:r,2:c);
% size(TmpTheta1)

[r,c] = size(Theta2);
TmpTheta2 = Theta2(1:r,2:c);
% size(TmpTheta2)

reg = (sum(sum(TmpTheta1.^2)) + sum(sum(TmpTheta2.^2))) * lambda / 2 / m;
% reg = (sum(sum(TmpTheta1.^2)) + sum(sum(TmpTheta2.^2))) * lambda / 2 / m;
J = J + reg;

% size(A3)
% size(Y)
Delta3 = A3 - Y;
% size(Theta2)
% size(Delta3)
dGdx = sigmoidGradient(X * Theta1');
[r,c] = size(dGdx);
dGdx = cat(2,zeros(r,1),dGdx);
% size(dGdx)
% size(Delta3*Theta2)
Delta2 = Delta3*Theta2.*dGdx;

Theta1_grad = Delta2'*X / m;
[r,c] = size(Theta1_grad);
Theta1_grad = Theta1_grad(2:r,1:c);
reg = lambda / m * Theta1;
reg(:,1) = 0;
Theta1_grad = Theta1_grad + reg;

Theta2_grad = Delta3'*A2 / m;
reg = lambda / m * Theta2;
reg(:,1) = 0;
Theta2_grad = Theta2_grad + reg;
% size(reg)
% [r,c] = size(Theta2_grad);
% Theta2_grad = Theta2_grad(1:r,2:c);
% size(Theta2_grad)



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
