function [BMC,PEB,DCM] = spm_dcm_bmc_peb(DCM,M,field)
% hierarchical (PEB) model comparison and selection
% FORMAT [BMC,PEB,DCM] = spm_dcm_bmc_peb(DCM,M,field)
%
% DCM    - {N [x M]} structure array of DCMs from N subjects
% ------------------------------------------------------------
%     DCM{i}.M.pE - prior expectation of parameters
%     DCM{i}.M.pC - prior covariances of parameters
%     DCM{i}.Ep   - posterior expectations
%     DCM{i}.Cp   - posterior covariance
%
% M.X    - second level design matrix, where X(:,1) = ones(N,1) [default]
% M.pE   - second level prior expectation of parameters
% M.pC   - second level prior covariances of parameters
% M.hE   - second level prior expectation of log precisions
% M.hC   - second level prior covariances of log precisions
%
% field  - parameter fields in DCM{i}.Ep to optimise [default: {'A','B'}]
%          'All' will invoke all fields (i.e. random effects)
%
% BMC    - Bayesian model comparison structure 
% -------------------------------------------------------------
%     BMC.F    - free energy over joint model space
%     BMC.P    - posterior probability over models
%     BMC.Px   - posterior probability over 1st level models
%     BMC.Pw   - posterior probability over 2nd level models
%     BMC.M    - second level model
%     BMC.K    - model space
%
% PEB    - selected second level model and parameter estimates
% -------------------------------------------------------------
%     PEB.Snames - string array of first level model names
%     PEB.Pnames - string array of parameters of interest
%     PEB.Pind   - indices of parameters in spm_vec(DCM{i}.Ep) 
%
%     PEB.M    -   first level (within subject) model
%     PEB.Ep   -   posterior expectation of second level parameters
%     PEB.Eh   -   posterior expectation of second level log-precisions
%     PEB.Cp   -   posterior covariance  of second level parameters
%     PEB.Ch   -   posterior covariance  of second level log-precisions
%     PEB.Ce   -   expected covariance of second level random effects
%     PEB.F    -   free energy of second level model
%
% DCM    - selected DCM structures with first level parameter estimates
% -------------------------------------------------------------------------
%
%--------------------------------------------------------------------------
% This routine performs Bayesian model comparison in the joint space of
% models specified in terms of (first level) model parameters and models
% specified in terms of (second level) group effects. The first level model
% space is defined by the columns of the DCM array, while the second level
% model space is specified by combinations of second level effects encoded 
% in a design matrix. first, the design matrix is a constant term or
% that models a group mean. This is assumed to be present a priori.
%
% this routine assumes that all the models have been reduced (i.e. inverted
% using Bayesian model reduction). It then use sempirical Bayes and the
% summary statistic approach tto evaluate the relative contributions of
% between subject effects by considering all combinations of columns in the
% design matrix.
%
% This Bayesian model comparison should be contrasted with model
% comparison at the second level. Here, we are interested in the best model
% of first level parameters that show a second level effect. This is not
% the same as trying to find the best model of second level effects. model
% comparison among second level parameters uses spm_dcm_bmr_peb.
%
% NB for EEG models the absence of a connection means it is equal to its
% prior mesn, not that is is zero.
%
% see also: spm_dcm_peb.m and spm_dcm_bmr_peb
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_dcm_peb_bmc.m 6306 2015-01-18 20:50:38Z karl $


% set up
%==========================================================================

% Number of subjects (data) and models (of those data)
%--------------------------------------------------------------------------
[Ns,Nm]   = size(DCM);

% % second level model
%--------------------------------------------------------------------------
if nargin < 2;
    M.X   = ones(Ns,1);
    M.pE  = DCM{1}.M.pE;
    M.pC  = DCM{1}.M.pC;
end
if ~isstruct(M)
    M = struct('X',M);
end

% use priors from the first level if necessary
%--------------------------------------------------------------------------
if ~isfield(M,'pE')
    M.pE = spm_vec(DCM{1}.M.pE);
end
if ~isfield(M,'pC')
    if isstruct(DCM{1}.M.pC)
        M.pC = diag(spm_vec(DCM{1}.M.pC));
    else
        M.pC = DCM{1}.M.pC;
    end
end
if isstruct(M.pE), M.pE =      spm_vec(M.pE) ; end
if isstruct(M.pC), M.pC = diag(spm_vec(M.pC)); end

% feels that specify which parameters are random effects
%--------------------------------------------------------------------------
if nargin < 3;
    field = {'A','B'};
end
if strcmpi(field,'all');
    field = fieldnames(DCM{1}.M.pE);
end
if ischar(field)
    field = {field};
end

% Bayesian model comparison in joint first and second level model space
%==========================================================================
K     = spm_perm_mtx(size(M.X,2));
K     = K(K(:,1),:);
Nk    = size(K,1);
Mk    = M;
for i = 1:Nm
    for k = 1:Nk
        
        % second level model reduction
        %------------------------------------------------------------------
        Mk.X     = M.X(:,K(k,:));
        PEB{i,k} = spm_dcm_peb(DCM(:,i),Mk,field);
        F(i,k)   = PEB{i,k}.F;
        
    end
end

% family wise inference over models and parameters
%==========================================================================
P      = F;
P(:)   = exp(P(:) - max(P(:)));
P(:)   = P/sum(P(:));

% family wise inference over mmodels and design
%--------------------------------------------------------------------------
Px     = sum(P,1);
Pw     = sum(P,2);

% select best empirical Bayes model
%--------------------------------------------------------------------------
[m i]     = max(Pw);
[m k]     = max(Px);
M.X       = M.X(:,K(k,:));
[PEB,DCM] = spm_dcm_peb(DCM(:,i),M,field);

% assemble BMC output structure
%--------------------------------------------------------------------------
BMC.M  = M;
BMC.F  = F;
BMC.P  = P;
BMC.Px = Px;
BMC.Pw = Pw;
BMC.K  = K;


% Show results
%==========================================================================
spm_figure('Getwin','PEB-BMC'); clf

subplot(3,2,1), [m i] = max(Pw); bar(Pw),
text(i - 1/4,m/2,sprintf('%-2.0f%%',m*100),'Color','w','FontSize',8)
title('First level','FontSize',16)
xlabel('Model','FontSize',12)
ylabel('Probability','FontSize',12)
axis([0 (Nm + 1) 0 1]), axis square

subplot(3,2,2), [m i] = max(Px); bar(Px),
text(i - 1/4,m/2,sprintf('%-2.0f%%',m*100),'Color','w','FontSize',8)
title('Second level','FontSize',16)
xlabel('Model','FontSize',12)
ylabel('Model probability','FontSize',12)
axis([0 (Nk + 1) 0 1]), axis square

subplot(3,2,3), imagesc(P')
title('Posterior probabilities','FontSize',16)
xlabel('Model (first level)','FontSize',12)
ylabel('Model (second level)','FontSize',12)
axis square

subplot(3,2,4), imagesc(M.X)
title('Design matrix','FontSize',16)
xlabel('Second level effect','FontSize',12)
ylabel('Subject','FontSize',12)
axis square

subplot(3,2,5), imagesc(F')
title('Free energy','FontSize',16)
xlabel('Model (first level)','FontSize',12)
ylabel('Model (second level)','FontSize',12)
axis square

subplot(3,2,6), imagesc(K)
title('Model space','FontSize',16)
xlabel('Second level effect','FontSize',12)
ylabel('Second level model','FontSize',12)
axis square

