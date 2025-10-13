Nts2 = linspace(0,100,1001);
N = length(Nts2); 

C_N=[];
H=0.99;

for i=3:N % compute covariance matrix
  for j=3:N
    ti=Nts2(i); tj=Nts2(j);
    % ti=i;tj=j;    
    C_N(i,j)=0.5*(ti^(2*H)+tj^(2*H)-abs(ti-tj)^(2*H));
  end
end
rng("shuffle");
[U,S]=eig(C_N); 
xsi = randn(N, 1); 
X_pre = U*(S^0.5);
X=U*(S^0.5)*xsi;

trans_noise = diff(X);
prom_noise = mean(trans_noise)
var_noise = std(trans_noise)