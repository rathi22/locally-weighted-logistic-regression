function y = lwlr(X_train, y_train, x, tau)

[m,n] = size(X_train);
theta = zeros(n,1);
D = zeros(m,m);
lambda = 0.0001;
eps = 0.000001;
w = zeros(m,1);
for i=1:m,
  w(i,1) = exp(-((X_train(i,1)-x(1))^2 + (X_train(i,2)-x(2))^2)/(2*tau*tau));
end;
while true,
  h = 1 ./ (1+exp(-X_train*theta));
  z = w.*(y_train-h);
  del = X_train'*z - lambda*theta;
  H = -X_train'*diag(w.*h.*(1-h))*X_train - lambda*eye(n);
  difference = inv(H)*del;
  if norm(difference) < eps,
    break;
  end;  
  theta = theta - difference;   
end;

y = theta'*x > 0;