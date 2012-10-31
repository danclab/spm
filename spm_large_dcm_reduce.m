function [DCM] = spm_large_dcm_reduce(DCM)
% Optimises the number of prior connectivity eigenmodes
% FORMAT [DCM] = spm_large_dcm_reduce (DCM)
%   DCM - DCM structure or its filename
%
% This routine optimises the number of eigenmodes of the prior covariance
% matrix using the eigenvectors of the functional connectivity matrix. The
% optimisation uses post hoc model reduction.
%__________________________________________________________________________
% Copyright (C) 2002-2012 Wellcome Trust Centre for Neuroimaging
 
% Karl Friston
% $Id: spm_large_dcm_reduce.m 5022 2012-10-30 19:25:02Z karl $
 
 
% create priors
%==========================================================================
 
 
% eigenvector constraints on pC for large models
%--------------------------------------------------------------------------
n     = size(DCM.M.pE.A,1);
j     = 1:(n*n);
 
% remove confounds and find principal modes
%--------------------------------------------------------------------------
y     = DCM.Y.y - DCM.Y.X0*(pinv(DCM.Y.X0)*DCM.Y.y);
V     = spm_svd(y');

for i = 1:n
    
    % remove minor modes from priors on A
    %----------------------------------------------------------------------
    v       = V(:,1:i);
    v       = kron(v*v',v*v');
    
    pc      = DCM.M.pC;
    pc(j,j) = v*DCM.M.pC(j,j)*v';
    rC{i}   = pc;
    
end
 
%-Loop over prior covariances to get log-evidences
%==========================================================================
 
% Get priors and posteriors
%--------------------------------------------------------------------------
qE    = DCM.Ep;
qC    = DCM.Cp;
pE    = DCM.M.pE;
pC    = DCM.M.pC;
 
% Remove (a priori) null space
%--------------------------------------------------------------------------
U     = spm_svd(pC);
qE    = U'*spm_vec(qE);
pE    = U'*spm_vec(pE);
qC    = U'*qC*U;
pC    = U'*pC*U;
 
% model search
%--------------------------------------------------------------------------
for i = 1:n
    S(i) = spm_log_evidence(qE,qC,pE,pC,pE,U'*rC{i}*U);
end
 
% model evidence
%--------------------------------------------------------------------------
S     = S - min(S);
p     = exp(S - max(S));
p     = p/sum(p);
 
% Show results
% -------------------------------------------------------------------------
spm_figure('Getwin','Graphics');
 
subplot(2,2,1)
bar(S)
title('log-posterior','FontSize',16)
xlabel('number of prior eigenmodes','FontSize',12)
ylabel('log-probability','FontSize',12)
set(gca,'YLim',[(max(S) - 1000), max(S)])
axis square
 
subplot(2,2,2)
bar(p)
title('model posterior','FontSize',16)
xlabel('number of prior eigenmodes','FontSize',12)
ylabel('probability','FontSize',12)
axis square
drawnow
 
%-Get posterior density of reduced model
%==========================================================================
 
% Get full priors and posteriors
%--------------------------------------------------------------------------
[p,i] = max(p);
qE    = DCM.Ep;
qC    = DCM.Cp;
pE    = DCM.M.pE;
pC    = DCM.M.pC;


[F,Ep,Cp] = spm_log_evidence_reduce(qE,qC,pE,pC,pE,rC{i});
 
% Bayesian inference and variance
%--------------------------------------------------------------------------
T        = full(spm_vec(pE));
Pp       = spm_unvec(1 - spm_Ncdf(T,abs(spm_vec(Ep)),diag(Cp)),Ep);
Vp       = spm_unvec(diag(Cp),Ep);
 
% Store parameter estimates
%--------------------------------------------------------------------------
DCM.M.pC = rC{i};
DCM.Ep   = Ep;
DCM.Cp   = Cp;
DCM.Pp   = Pp;
DCM.Vp   = Vp;
 
% and in DEM format
%--------------------------------------------------------------------------
DCM.qP.P{1} = Ep;
DCM.qP.C    = Cp;
DCM.qP.V{1} = spm_unvec(diag(Cp),Ep);
 
% approximations to model evidence: negative free energy, AIC, BIC
%--------------------------------------------------------------------------
DCM.F    = F;
evidence = spm_dcm_evidence(DCM);
DCM.AIC  = evidence.aic_overall;
DCM.BIC  = evidence.bic_overall;


