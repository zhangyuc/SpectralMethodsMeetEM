Nround = 10;
mode = 0;

error1_predict = zeros(1,Nround);
error2_predict = zeros(1,Nround);

% Load datasets
% To test other datasets, replace bluebird_crowd 
% and bluebird_truth by the names of other data files
A = importdata('bluebird_crowd.txt');
B = importdata('bluebird_truth.txt');

% ====================================================
n = max(A(:,1));
m = max(A(:,2));
k = max(B(:,2));

y = zeros(n,1);
for i = 1:size(B,1)
    y(B(i,1)) = B(i,2);
end
valid_index = find(y > 0);

Z = zeros(n,k,m);
for i = 1:size(A,1)
    Z(A(i,1),A(i,3),A(i,2)) = 1;
end

%===================== majority vote ================
q = mean(Z,3);
[I J] = max(q');
accuracy = 0;
for j = 1:n
    maxq = max(q(j,:));
    if y(j) > 0 && q(j,y(j)) == maxq
        accuracy = accuracy + 1 / size(find(q(j,:) == maxq), 2);
    end
end
error_majority_vote = 1 - accuracy / size(valid_index,1)

%===================== Sewoung Oh ================

t = zeros(n,k-1);
for l = 1:k-1
    U = zeros(n,m);
    for i = 1:size(A,1)
        U(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    
    B = U - ones(n,1)*(ones(1,n)*U)/n;
    [U S V] = svd(B);
    u = U(:,1);
    v = V(:,1);
    u = u / norm(u);
    v = v / norm(v);
    pos_index = find(v>=0);
    if sum(v(pos_index).^2) >= 1/2
        t(:,l) = sign(u);
    else
        t(:,l) = -sign(u);
    end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_KOS = mean(y(valid_index) ~= (J(valid_index)))
% 
%===================== Ghosh-SVD ================

t = zeros(n,k-1);
for l = 1:k-1
    O = zeros(n,m);
    for i = 1:size(A,1)
        O(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    
    [U S V] = svd(O);
    u = sign(U(:,1));
    if u'*sum(O,2) >= 0
        t(:,l) = sign(u);
    else
        t(:,l) = -sign(u);
    end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_GhostSVD = mean(y(valid_index) ~= (J(valid_index)))

% %===================== Ratio of Eigenvalues ================

t = zeros(n,k-1);
for l = 1:k-1
    O = zeros(n,m);
    for i = 1:size(A,1)
        O(A(i,1),A(i,2)) = 2*(A(i,3) > l)-1;
    end
    G = abs(O);
    
        % ========== algorithm 1 =============
%         [U S V] = svd(O'*O);
%         v1 = U(:,1);
%         [U S V] = svd(G'*G);
%         v2 = U(:,1);
%         v1 = v1./v2;
%         u = O*v1;
        % ========== algorithm 2 =============
        R1 = (O'*O)./(G'*G+10^-8);
        R2 = (G'*G > 0)+1-1;
        [U S V] = svd(R1);
        v1 = U(:,1);
        [U S V] = svd(R2);
        v2 = U(:,1);
        v1 = v1./v2;
        u = O*v1;
        
        if u'*sum(O,2) >= 0
            t(:,l) = sign(u);
        else
            t(:,l) = -sign(u);
        end
end

J = ones(n,1)*k;
for j = 1:n
    for l = 1:k-1
        if t(j,l) == -1
            J(j) = l;
            break;
        end
    end
end
error_RatioEigen = mean(y(valid_index) ~= (J(valid_index)))

% 
%===================== EM with majority vote ================
q = mean(Z,3);
q = q ./ repmat(sum(q,2),1,k);
mu = zeros(k,k,m);

% EM update
for iter = 1:Nround
    for i = 1:m
        mu(:,:,i) = (Z(:,:,i))'*q;
        mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
        
        for c = 1:k
            mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
        end
    end
    


    q = zeros(n,k);
    for j = 1:n
        for c = 1:k
            for i = 1:m
                if Z(j,:,i)*mu(:,c,i) > 0
                    q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
                end
            end
        end
        q(j,:) = exp(q(j,:));
        q(j,:) = q(j,:) / sum(q(j,:));
    end

    [I J] = max(q');
    error1_predict(iter) = mean(y(valid_index) ~= (J(valid_index))');
end
error1_predict


%===================== EM with spectral method ==============
% method of moment
group = mod(1:m,3)+1;
Zg = zeros(n,k,3);
cfg = zeros(k,k,3);
for i = 1:3
    I = find(group == i);
    Zg(:,:,i) = sum(Z(:,:,I),3);
end

x1 = Zg(:,:,1)';
x2 = Zg(:,:,2)';
x3 = Zg(:,:,3)';

muWg = zeros(k,k+1,3);
muWg(:,:,1) = SolveCFG(x2,x3,x1);
muWg(:,:,2) = SolveCFG(x3,x1,x2);
muWg(:,:,3) = SolveCFG(x1,x2,x3);

mu = zeros(k,k,m);
for i = 1:m
    x = Z(:,:,i)';
    x_alt = sum(Zg,3)' - Zg(:,:,group(i))';
    muW_alt = (sum(muWg,3) - muWg(:,:,group(i)));
    mu(:,:,i) = (x*x_alt'/n) / (diag(muW_alt(:,k+1)/2)*muW_alt(:,1:k)');
    
    mu(:,:,i) = max( mu(:,:,i), 10^-6 );
    mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
    for j = 1:k
        mu(:,j,i) = mu(:,j,i) / sum(mu(:,j,i));
    end
end

% EM update
for iter = 1:Nround
    q = zeros(n,k);
    for j = 1:n
        for c = 1:k
            for i = 1:m
                if Z(j,:,i)*mu(:,c,i) > 0
                    q(j,c) = q(j,c) + log(Z(j,:,i)*mu(:,c,i));
                end
            end
        end
        q(j,:) = exp(q(j,:));
        q(j,:) = q(j,:) / sum(q(j,:));
    end

    for i = 1:m
        mu(:,:,i) = (Z(:,:,i))'*q;
        
        mu(:,:,i) = AggregateCFG(mu(:,:,i),mode);
        for c = 1:k
            mu(:,c,i) = mu(:,c,i)/sum(mu(:,c,i));
        end
    end

    [I J] = max(q');
    error2_predict(iter) = mean(y(valid_index) ~= (J(valid_index))');
end
error2_predict

