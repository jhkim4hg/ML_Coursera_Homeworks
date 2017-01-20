% -------------------------------------------------------------

% Implement Part III -- Regularization with cost function/gradients
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% The formula for regularization is given on page 12 and is as
% follows: Delta(l(i,j)) = 1/m*delta(l(i,j)) + lambda/m*(Theta(l(i,j))
% for j >= 1

% Implement for Theta1 and Theta2 when l = 0
Theta1_grad(:,1) = Theta1_grad(:,1)./m;
Theta2_grad(:,1) = Theta2_grad(:,1)./m;

% Implement for Theta1 and Theta 2 when l > 0
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m + ( (lambda/m) * Theta1(:,2:end) );
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + ( (lambda/m) * Theta2(:,2:end) );


% -------------------------------------------------------------
