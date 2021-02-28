function [p0,X,F,f,NESS] = spm_ness_hd(M,x)
% nonequilibrium steady-state under a Helmholtz decomposition
% FORMAT [p0,X,F,f,NESS] = spm_ness_hd(M,x)
%--------------------------------------------------------------------------
% M   - model specification structure
% Required fields:
%    M.f   - dx/dt   = f(x,u,P)  {function string or m-file}
%    M.pE  - P       = parameters of equation of motion
%    M.x   - (n x 1) = x(0) = expansion point
%    M.W   - (n x n) - precision matrix of random fluctuations
% x    - cell array of vectors specifying evaluation grid
%
% p0      - nonequilibrium steady-state
% X       - evaluation points of state space
% F       - expected flow
% f       - original flow
%
% NESS.H  - expected Hessian
% NESS.J  - expected Jacobian
% NESS.E  - Lyapunov exponents
% NESS.H2 - expected Euclidean norm of Hessian
% NESS.J2 - expected Euclidean norm of Jacobian
% NESS.D2 - correlation dimension
% NESS.bS - p0 = spm_softmax(spm_dctmtx(nx,nb)*bS);
% NESS.nb - number of basis functions
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_ness_hd.m 8071 2021-02-28 16:12:22Z karl $


% event space: get or create X - coordinates of evaluation grid
%--------------------------------------------------------------------------
if nargin > 1
    U = spm_ness_U(M,x);                    % get state space and flow
else
    U = spm_ness_U(M);                      % get state space from M.X
end
X  = U.X;                                   % sample points
f  = U.f;                                   % flow
J  = U.J;                                   % Jacobian
G  = U.G;                                   % flow operator (symmetric)

% size of subspace (nx) and probability bin size (dx)
%--------------------------------------------------------------------------
[nX,n] = size(X);

% gradients based upon Helmholtz decomposition
%==========================================================================

% flow operator Q = (R + G) under the assumption dQ/dx = 0
%--------------------------------------------------------------------------
I     = speye(n,n);                         % identity matrix
Q     = zeros(n,n,nX);                      % flow operator
d     = speye(n*n)/512;                     % regulariser
for i = 1:nX
    
    % solenoidal operator (R = Q(:,:,i))
    %----------------------------------------------------------------------
    JJ       = kron(I,J(:,:,i)) + kron(conj(J(:,:,i)),I);
    Y        = spm_vec(J(:,:,i)*G - G*J(:,:,i)');
    Q(:,:,i) = reshape((JJ'*JJ + d)\(JJ'*Y),n,n);

end


% deal with the correction term by minimising |(f - L) - Q*dS/dx|
%==========================================================================
% where f = Q*dS/dx - L
%         = Q*D*S   - L
% subject to Q = Q + dQ: dQ = -qQ'
%--------------------------------------------------------------------------
b     = U.b;                     % orthonormal polynomial basis (S)
c     = U.c;                     % orthonormal polynomial basis (Q)
D     = U.D;                     % derivative operator (S)
V     = U.V;                     % derivative operator (Q)
dQdp  = U.dQdp;                  % gradients of Q  w.r.t. flow parameters
dbQdp = U.dbQdp;                 % gradients of bQ w.r.t. flow parameters
dLdp  = U.dLdp;                  % gradients of L  w.r.t. flow parameters
bG    = U.bG;                    % symmetric part of flow operator

% precision of parameters
%--------------------------------------------------------------------------
nb    = size(b,2);               % number of coefficients for S
nc    = size(c,2);               % number of coefficients for Q
nB    = numel(dQdp);             % number of coefficients for flow operator
nE    = exp(16);                 % initial error norm

Ib    = speye(nb,nb)*exp(-16);   % prior precision Sp
IB    = speye(nB,nB)*exp(-16);   % prior precision Qp

% initialise parameters Qp of flow operator
%--------------------------------------------------------------------------
bQ    = zeros(n,n,nc);
for i = 1:n
    for j = 1:n
        bQ(i,j,:) = c'*squeeze(Q(i,j,:));
    end
end
Qp    = zeros(nB,1);
for i = 1:nB
    Qp(i) = dbQdp{i}(:)'*bQ(:);
end
    
% iterated least-squares to estimate flow operator
%==========================================================================
dEdp  = zeros(nX*n,nB);                   % error gradients
D     = spm_cat(D);                       % derivative operator
for k = 1:256
     
    % solenoidal operator
    %----------------------------------------------------------------------
    bQ    = 0;
    for i = 1:nB
        bQ = bQ + dbQdp{i}*Qp(i);
    end
    
    % correction term for solenoidal flow L and Kroneckor form of Q
    %----------------------------------------------------------------------    
    Q     = cell(n,n);
    L     = zeros(nX,n);
    for i = 1:n
        for j = 1:n
            bQij   = squeeze(bQ(i,j,:) + bG(i,j,:));
            L(:,i) = L(:,i) - V{j}*bQij;
            Q{i,j} = spdiags(c*bQij,0,nX,nX);
        end
    end
    Q   = spm_cat(Q);
        
    % potential gradients f = Q*D*b*b'*S - L
    %----------------------------------------------------------------------
    QD  = Q*D;
    Y   = spm_vec(f' + L);
    Sp  = (QD'*QD + Ib)\(QD'*Y);
    
    % dEdb  = dLdb - dQdb*D*b*b'*S;
    %----------------------------------------------------------------------
    DS    = D*Sp;
    for i = 1:nB
        dEdp(:,i) = dQdp{i}*DS - spm_vec(dLdp{i});
    end
    
    % dEdb'*dEdb*db = dEdb'*E: E = f + b*L - Q*Dx*b*b'*S
    %----------------------------------------------------------------------
    E     = Y  - QD*Sp;
    Qp    = Qp + (dEdp'*dEdp + IB)\(dEdp'*E);
    
    % report
    %----------------------------------------------------------------------
    if (nE - norm(E)) < 1/64
        break
    end
    nE    = norm(E);
    fprintf('%-6s: %i %-6.3e\n','NESS',k,nE)
    
end

% evaluate Q from basis function coefficients Qb
%--------------------------------------------------------------------------
Q     = zeros(n,n,nX);
for i = 1:n
    for j = 1:n
        Q(i,j,:) = c*reshape(bQ(i,j,:),nc,1);
    end
end

% NESS density
%--------------------------------------------------------------------------
p0    = spm_softmax(b*Sp);              

% predicted flow: F   = Q*Dx*S - L
%--------------------------------------------------------------------------
F     = reshape(QD*Sp,nX,n) - L;

% Hessian D*D*S
%--------------------------------------------------------------------------
H     = zeros(n,n,nX);
HH    = cell(n,n);
for i = 1:n
    for j = 1:n
       HH{i,j} = U.H{i,j}*Sp;    
    end
end
for i = 1:n
    for j = 1:n
        for k = 1:nX
            H(i,j,k) = HH{i,j}(k);
        end
    end
end

% eigenvalues of Jacobian
%--------------------------------------------------------------------------
E     = zeros(n,nX);
for i = 1:nX
    E(:,i)   = sort(eig(J(:,:,i)),'descend','ComparisonMethod','real');
end

% parameters of log NESS density Sp and flow operator Qp
%--------------------------------------------------------------------------
Sp    = U.v*Sp;
Qp    = U.u*Qp;
i     = abs(Sp) < exp(-16);
Sp(i) = 0;
i     = abs(Qp) < exp(-16);
Qp(i) = 0;
Ep.Sp = Sp;
Ep.Qp = Qp;

% assemble NESS structure
%--------------------------------------------------------------------------
NESS.H  = spm_dot(H   ,p0);               % expected Hessian
NESS.J  = spm_dot(J   ,p0);               % expected Jacobian
NESS.E  = spm_dot(E   ,p0);               % Lyapunov exponents
NESS.H2 = spm_dot(H.^2,p0);               % expected Euclidean norm of Hessian
NESS.J2 = spm_dot(J.^2,p0);               % expected Euclidean norm of Jacobian
NESS.D2 = 2 + abs(E(1) + E(2))/abs(E(3)); % correlation dimension
NESS.Ep = Ep;                             % parameters of flow

% reshape nonequilibrium steady-state density
%--------------------------------------------------------------------------
% p0    = spm_softmax(spm_polymtx(x,nb)*Ep.Sp);
p0      = reshape(p0,U.nx);

return
