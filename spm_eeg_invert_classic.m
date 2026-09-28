function [D] = spm_eeg_invert_classic(D,val)
% Parallelized version of spm_eeg_invert_classic
% This version processes multiple time windows (wois) in parallel

Nl = length(D);

if Nl>1
    error('function only defined for a single subject');
end

if nargin > 1
    D.val = val;
elseif ~isfield(D, 'val')
    D.val = 1;
end

val=D.val;
inverse = D.inv{val}.inverse;

try, type = inverse.type;   catch, type = 'GS';     end
try, s    = inverse.smooth; catch, s    = 0.6;      end
try, Np   = inverse.Np;     catch, Np   = 256;      end
try, Nr   = inverse.Nr;     catch, Nr   = 16;       end
try, xyz  = inverse.xyz;    catch, xyz  = [0 0 0];  end
try, rad  = inverse.rad;    catch, rad  = 128;      end
try, hpf  = inverse.hpf;    catch, hpf  = 48;       end
try, lpf  = inverse.lpf;    catch, lpf  = 0;        end
try, sdv  = inverse.sdv;    catch, sdv  = 4;        end
try, Han  = inverse.Han;    catch, Han  = 1;        end
try, woi  = inverse.woi;    catch, woi  = [];       end
try, Nm   = inverse.Nm;     catch, Nm   = [];       end
try, Nt   = inverse.Nt;     catch, Nt   = [];       end
try, Ip   = inverse.Ip;     catch, Ip   = [];       end
try, QE    = inverse.QE;     catch,  QE=1;          end
try, Qe0   = inverse.Qe0;     catch, Qe0   = exp(-5);       end
try, inverse.A;     catch, inverse.A   = [];       end
try, SHUFFLELEADS=inverse.SHUFFLELEADS;catch, SHUFFLELEADS=0;end
try, nlayers = inverse.nlayers; catch, nlayers = 1; end

type = inverse.type;

modalities = D.inv{val}.forward.modality;

Nmax  = 16;

fprintf('Checking leadfields')
[L,D] = spm_eeg_lgainmat(D);
Nd=size(L,2);

if ~isempty(Ip)
    Np   = length(Ip);
else
    Ip=ceil([1:Np]*Nd/Np);
end

persistent permind;

rand(2)
if SHUFFLELEADS
    rng('shuffle')
    if isempty(permind)
        permind=randperm(size(L,1));
    end
    L=L(permind,:);
    warning('PERMUTING LEAD FIELDS !');
    permind(1:3)
end

if size(modalities,1)>1
    error('not defined for multiple modalities');
end
if strcmp(modalities,'MEG')
    Ic  = setdiff(...
            union(...
                D.indchantype('MEG'),...
                D.indchantype('MEGPLANAR')...
            ),...
            badchannels(D)...
          );
else
    Ic  = setdiff(D.indchantype(modalities), badchannels(D));
end
Nd    = size(L,2);

fprintf(' - done\n')

if s>=1
    smoothtype='mesh_smooth';
else
    smoothtype='msp_smooth';
end
if s<0
    smoothtype='mm_smooth';
    s=-s;
end
vert  = D.inv{val}.mesh.tess_mni.vert;
face  = D.inv{val}.mesh.tess_mni.face;
M1.faces=face;
M1.vertices=vert;

switch smoothtype
    case 'mesh_smooth'
        fprintf('Using SPM smoothing for priors:')
        
        Qi    = speye(Nd,Nd);
        [QG]=spm_mesh_smooth(M1,Qi,round(s));
        QG    = QG.*(QG > exp(-8));
        
        QG    = QG*QG;
        disp('Normalising smoother');
        QG=QG./repmat(sum(QG,2),1,size(QG,1));
        
    case 'msp_smooth'
        fprintf('Computing Green function from graph Laplacian to smooth priors:')
        
        A     = spm_mesh_distmtx(struct('vertices',vert,'faces',face),0);
        
        GL    = A - spdiags(sum(A,2),0,Nd,Nd);
        GL    = GL*s/2;
        Qi    = speye(Nd,Nd);
        QG    = sparse(Nd,Nd);
        
        for i = 1:8
            QG = QG + Qi;
            Qi = Qi*GL/i;
        end
        
        QG    = QG.*(QG > exp(-8));
        QG    = QG*QG;
        
    case 'mm_smooth'
        
         kernelname=spm_eeg_smoothmesh_mm(D.inv{val}.mesh.tess_ctx,s);
         asmth=load(kernelname,'QG','M','faces');
         if isfield(asmth, 'M')
             if isa(asmth.M, 'gifti')
                 meshFaces = double(asmth.M.faces);
             else
                 error('M is not a recognized format.');
             end
         elseif isfield(asmth, 'faces')
             meshFaces = double(asmth.faces);
         else
             error('No valid mesh face data found in the smoothing kernel file.');
         end
         if max(int32(meshFaces)-int32(face))~=0
             error('Smoothing kernel used different mesh');
         end
         QG=asmth.QG;
         clear asmth;
          
end

clear Qi A GL
fprintf(' - done\n')

fprintf('Optimising and aligning spatial modes ...\n')

if isempty(inverse.A)
    if isempty(Nm)
        [U,ss,vv]    = spm_svd((L*L'),exp(-16));
        A     = U';
        UL    = A*L;
        
    else
        [U,ss,vv]    = spm_svd((L*L'),0);
        if length(ss)<Nm
            disp('number available');
            length(ss)
            error('Not this many spatial modes in lead fields');
        end
        
        ss=ss(1:Nm);
        disp('using preselected number spatial modes !');
        A     = U(:,1:Nm)';
        UL    = A*L;
    end
else
    disp('Using pre-specified spatial modes');
    if isempty(Nm)
        error('Need to specify number of spatial modes if U is prespecified');
    end
    A=inverse.A;
    UL=A*L;
end

Nm    = size(UL,1);

clear ss vv

fprintf('Using %d spatial modes',Nm)

Is    = 1:Nd;
Ns    = length(Is);

F=zeros(1,size(woi,1));

if isempty(woi)
    woi = 1000*[min(D.time) max(D.time)];
end
R2=zeros(1,size(woi,1));
VE=zeros(1,size(woi,1));

if size(woi,1)==1 || ~spm_get_defaults('use_parfor')
    for w_idx=1:size(woi,1)
        [F(w_idx), R2(w_idx), VE(w_idx), J_temp{w_idx}, M_temp{w_idx}, ...
         Cq_temp{w_idx}, U_temp{w_idx}, V_temp{w_idx}, Vq_temp{w_idx}, ...
         S_temp{w_idx}, It_temp{w_idx}, Ik_temp{w_idx}, ID_temp{w_idx}, ...
         pst_temp{w_idx}, dct_temp{w_idx}, EBBlayer_diag_temp{w_idx}] = ...
            process_woi(D, val, w_idx, woi, A, UL, QG, Ns, Ip, Np, QE, ...
                        Qe0, type, Nmax, Nt, Nr, Han, lpf, hpf, sdv, Ic, ...
                        vert, nlayers);
    end
else
    % Create a parallel pool if one doesn't exist
    if isempty(gcp('nocreate'))
        parpool;
    end

    % Process each woi in parallel
    parfor w_idx=1:size(woi,1)
        [F(w_idx), R2(w_idx), VE(w_idx), J_temp{w_idx}, M_temp{w_idx},...
         Cq_temp{w_idx}, U_temp{w_idx}, V_temp{w_idx}, Vq_temp{w_idx},...
         S_temp{w_idx}, It_temp{w_idx}, Ik_temp{w_idx}, ID_temp{w_idx},...
         pst_temp{w_idx}, dct_temp{w_idx}, EBBlayer_diag_temp{w_idx}] = ...
            process_woi(D, val, w_idx, woi, A, UL, QG, Ns, Ip, Np, QE,...
                        Qe0, type, Nmax, Nt, Nr, Han, lpf, hpf, sdv, Ic,...
                        vert, nlayers);
    end
end

if size(woi,1)>1
    inverse.M_win={};
end
% Combine results from parallel processing
for w_idx=1:size(woi,1)
    if w_idx == 1
        inverse.type   = type;
        inverse.smooth = s;
        inverse.M      = M_temp{w_idx};
        inverse.J      = J_temp{w_idx};
        inverse.L      = UL;
        inverse.qC     = Cq_temp{w_idx};
        inverse.tempU  = U_temp{w_idx};
        inverse.V      = V_temp{w_idx};
        inverse.qV     = Vq_temp{w_idx};
        inverse.T      = S_temp{w_idx};
        inverse.U      = {A};
        inverse.Is     = Is;
        inverse.It     = It_temp{w_idx};
        inverse.Ik     = Ik_temp{w_idx};
        try
            inverse.Ic{1} = Ic;
        catch
            inverse.Ic = Ic;
        end
        inverse.Nd     = Nd;
        inverse.pst    = pst_temp{w_idx};
        inverse.dct    = dct_temp{w_idx};
        inverse.ID     = ID_temp{w_idx};
    end
    inverse.F(w_idx)   = F(w_idx);
    inverse.R2(w_idx)  = R2(w_idx);
    inverse.VE(w_idx)  = R2(w_idx).*VE(w_idx);
    if size(woi,1)>1
        inverse.M_win{w_idx}=M_temp{w_idx};
    end
    if strcmp(type, 'EBBlayer')
        if w_idx == 1
            inverse.EBBlayer_diag = EBBlayer_diag_temp{w_idx};
        end
        if size(woi,1) > 1
            inverse.EBBlayer_diag_win{w_idx} = EBBlayer_diag_temp{w_idx};
        end
    end
end
inverse.woi    = woi;
inverse.Ip     = Ip;
inverse.modality = modalities;

D.inv{val}.inverse = inverse;
D.inv{val}.method  = 'Imaging';

if ~spm('CmdLine')
    spm_eeg_invert_display(D);
    drawnow
end

end

function [F_out, R2_out, VE_out, J_out, M_out, Cq_out, U_out, V_out, Vq_out, ...
          S_out, It_out, Ik_out, ID_out, pst_out, dct_out, EBBlayer_diag_out] = ...
    process_woi(D, val, w_idx, woi, A, UL, QG, Ns, Ip, Np, QE, Qe0, ...
                type, Nmax, Nt, Nr, Han, lpf, hpf, sdv, Ic, vert, nlayers)

% This function processes a single time window of interest (woi)
w = woi(w_idx,:);

if ~isempty(Ip)
    Np = length(Ip);
else
    Ip = ceil([1:Np]*Ns/Np);
end

EBBlayer_diag_out = struct();

% =========================================================================
% Optional EBBlayer pair-score diagnostic vertex
% =========================================================================
EBBlayer_diag_vertex = [];

if strcmp(type, 'EBBlayer')
    try
        EBBlayer_diag_vertex = D.inv{val}.inverse.EBBlayer_diag_vertex;
    catch
        EBBlayer_diag_vertex = [];
    end

    if ~isempty(EBBlayer_diag_vertex)
        if ~isscalar(EBBlayer_diag_vertex) || ...
                ~isfinite(EBBlayer_diag_vertex) || ...
                EBBlayer_diag_vertex < 1 || ...
                EBBlayer_diag_vertex ~= round(EBBlayer_diag_vertex)
            error( ...
                ['EBBlayer_diag_vertex must be a positive one-based ' ...
                 'within-layer vertex index.'] ...
            );
        end
        EBBlayer_diag_vertex = double(EBBlayer_diag_vertex);
    end
end

% =========================================================================
% EBBlayer pair-selection parameters
%
% The current EBBlayer algorithm always uses independent TOP-K selection
% of SUM and DIFF hypotheses at each cortical column.
%
%   EBBlayer_sum_pair_topk
%       Number of highest-scoring genuine interior SUM hypotheses retained
%       per cortical column.
%
%   EBBlayer_diff_pair_topk
%       Number of highest-scoring DIFF hypotheses retained per cortical
%       column.
%
% Setting K equal to the total number of layer pairs is equivalent to keeping
% all eligible pair hypotheses.  This provides a simple way to test TOP-K
% sensitivity without retaining the older ALL / SINGLE / ANCHOR modes.
%
% SUM within-pair mixing is always optimized continuously over
%
%       q+(r) = qA + r*qB,   r >= 0,
%
% with endpoint optima (r=0 or r=Inf) excluded because they are single-layer
% hypotheses already represented by IND.
%
% DIFF remains the fixed equal-magnitude qA-qB hypothesis.
% =========================================================================
EBBlayer_sum_pair_topk  = 2;
EBBlayer_diff_pair_topk = 2;

if strcmp(type, 'EBBlayer')
    try
        EBBlayer_sum_pair_topk = D.inv{val}.inverse.EBBlayer_sum_pair_topk;
    catch
        EBBlayer_sum_pair_topk = 2;
    end

    try
        EBBlayer_diff_pair_topk = D.inv{val}.inverse.EBBlayer_diff_pair_topk;
    catch
        EBBlayer_diff_pair_topk = 2;
    end

    if ~isscalar(EBBlayer_sum_pair_topk) || ...
            ~isfinite(EBBlayer_sum_pair_topk) || ...
            EBBlayer_sum_pair_topk < 1 || ...
            EBBlayer_sum_pair_topk ~= round(EBBlayer_sum_pair_topk)
        error('EBBlayer_sum_pair_topk must be a positive integer.');
    end
    EBBlayer_sum_pair_topk = double(EBBlayer_sum_pair_topk);

    if ~isscalar(EBBlayer_diff_pair_topk) || ...
            ~isfinite(EBBlayer_diff_pair_topk) || ...
            EBBlayer_diff_pair_topk < 1 || ...
            EBBlayer_diff_pair_topk ~= round(EBBlayer_diff_pair_topk)
        error('EBBlayer_diff_pair_topk must be a positive integer.');
    end
    EBBlayer_diff_pair_topk = double(EBBlayer_diff_pair_topk);
end

AY    = {};
AYYA  = 0;

It = (w/1000 - D.timeonset)*D.fsample + 1;
It = max(1,It(1)):min(It(end), length(D.time));
It = fix(It);
disp(sprintf('Number of samples %d',length(It)));

pst = 1000*D.time;
pst = pst(It);
dur = (pst(end) - pst(1))/1000;
dct = (It - It(1))/2/dur;
Nb  = length(It);

K   = exp(-(pst - pst(1)).^2/(2*sdv^2));
K   = toeplitz(K);
qV  = sparse(K*K');

T   = spm_dctmtx(Nb,Nb);

j   = find((dct >= lpf) & (dct <= hpf));
T   = T(:,j);
dct = dct(j);

if Han
    W = sparse(1:Nb,1:Nb,spm_hanning(Nb));
else
    W = 1;
end

try
    trial = D.inv{D.val}.inverse.trials;
catch
    trial = D.condlist;
end
Ntrialtypes = length(trial);

YY = 0;
N  = 0;

badtrialind = D.badtrials;
Ik = [];
for j = 1:Ntrialtypes
    c = D.indtrial(trial{j});
    [c1,ib] = intersect(c,badtrialind);
    c = c(setxor(1:length(c),ib));
    Ik = [Ik c];
    Nk = length(c);
    for k = 1:Nk
        Y = A*D(Ic,It,c(k));
        YY = YY + Y'*Y;
        N = N + 1;
    end
end
YY = YY./N;

YY = W'*YY*W;
YTY = T'*YY*T;

if isempty(Nt)
    [U, E] = spm_svd(YTY,exp(-8));
    if isempty(U)
        warning('nothing found using spm svd, using svd');
        [U E] = svd(YTY);
    end
    E = diag(E)/trace(YTY);
    Nr = min(length(E),Nmax);
    Nr = max(Nr,1);
else
    [U, E] = svd(YTY);
    E = diag(E)/trace(YTY);
    disp('Fixed number of temporal modes');
    Nr = Nt;
end

V = U(:,1:Nr);
VE_out = sum(E(1:Nr));

fprintf('Using %i temporal modes, ',Nr)
fprintf('accounting for %0.2f percent average variance\n',full(100*VE_out))

S = T*V;
Vq = S*pinv(S'*qV*S)*S';

UYYU = 0;
AYYA = 0;
Nn = 0;
AY = {};
Ntrials = 0;

for j = 1:Ntrialtypes
    UY{j} = sparse(0);
    c = D.indtrial(trial{j});
    [c1,ib] = intersect(c,badtrialind);
    c = c(setxor(1:length(c),ib));
    Nk = length(c);
    
    for k = 1:Nk
        Y = D(Ic,It,c(k))*S;
        Y = A*Y;
        
        Nn = Nn + Nr;
        
        YY = Y*Y';
        Ntrials = Ntrials+1;
        
        UY{j} = UY{j} + Y;
        UYYU = UYYU + YY;
        
        AY{end + 1} = Y;
        AYYA = AYYA + YY;
    end
end

AY = spm_cat(AY);

ID = spm_data_id(AY);

AQeA = A*QE*A';
Qe{1} = AQeA/(trace(AQeA));

Q0 = Qe0*trace(AYYA)*Qe{1}./sum(Nn);

allind = [];
switch(type)
    case {'MSP','GS','ARD'}
        Qp = {};
        LQpL = {};
        for i = 1:Np
            q = QG(:,Ip(i));
            Qp{end + 1}.q = q;
            LQpL{end + 1}.q = UL*q;
        end
        
    case {'EBB'}
        disp('NB smooth EBB algorithm !');
        InvCov = spm_inv(AYYA);
        allsource = sparse(Ns,1);
        Sourcepower = sparse(Ns,1);
        for bk = 1:Ns
            q = QG(:,bk);
            
            smthlead = UL*q;
            if ~all(smthlead==0)
                normpower = 1/(smthlead'*smthlead);
                Sourcepower(bk) = 1/(smthlead'*InvCov*smthlead);
                allsource(bk) = Sourcepower(bk)./normpower;
            end
        end
        allsource = allsource/max(allsource);
        
        Qp{1} = diag(allsource);
        LQpL{1} = UL*diag(allsource)*UL';
    
    case {'EBBcorr'}
        disp('NB smooth correlated source EBB algorithm !');
        InvCov = spm_inv(AYYA);
        halfNs = ceil(Ns/2);
        allsource = sparse(Ns,1);
        Sourcepower = sparse(Ns,1);
        DualSourcepower = sparse(halfNs,1);
        alldualsource = sparse(Ns,1);
        for bk = 1:Ns
            q = QG(:,bk);
            
            smthlead = UL*q;
            normpower = 1/(smthlead'*smthlead);
            Sourcepower(bk) = 1/(smthlead'*InvCov*smthlead);
            allsource(bk) = Sourcepower(bk)./normpower;
        end
        
        leftbrainind = find(vert(:,1)<0);
        
        for bk = 1:length(vert)
            vertind = bk;
            
            reflectpos = [-vert(vertind,1) vert(vertind,2) vert(vertind,3)];
            d1 = vert - repmat(reflectpos,length(vert),1);
            dist1 = dot(d1',d1');
            [d1,reflectind] = min(dist1);
            q = QG(:,vertind)+QG(:,reflectind);
            smthlead = UL*q;
            normpower = 1/(smthlead'*smthlead);
            DualSourcepower(vertind) = 1/(smthlead'*InvCov*smthlead);
            alldualsource(vertind) = alldualsource(vertind)+DualSourcepower(vertind)./(normpower*4);
            alldualsource(reflectind) = alldualsource(reflectind)+DualSourcepower(vertind)./(normpower*4);
        end
        
        allsource = allsource+alldualsource;
        allsource = allsource/max(allsource);
        
        Qp{1} = diag(allsource);
        LQpL{1} = UL*diag(allsource)*UL';
      
    case {'EBBlayer'}

        disp('NB full-covariance laminar EBB algorithm !');

        % AYYA is sensor data covariance after projecting to the spatial 
        % and temporal modes
        InvCov = spm_inv(AYYA);

        % What counts as a value too small to care about
        tiny = 1e-30;

        % Multilayer mesh sanity check
        vert_per_layer = length(vert) / nlayers;
        if abs(vert_per_layer - round(vert_per_layer)) > eps
            error('length(vert) must be divisible by nlayers.');
        end
        vert_per_layer = round(vert_per_layer);
        V = vert_per_layer;
        fprintf('%d layers, %d vertices per layer\n', nlayers, V);

        % 1) Independent component
        % Standard smooth EBB prior.
        ind_source = sparse(Ns,1);

        % Iterate through spatial modes
        for bk = 1:Ns
            % QG = spatial smoothing / patch basis on the source mesh
            % tells EBB what spatial source pattern is associated with 
            % putting activity at a particular mesh vertex (taking into
            % account that it is a smoothed patch). This is why it's
            % important that there are no edges connected vertices across
            % layers in the multilayer mesh
            % q = spatial patch centred on source vertex bk
            q = QG(:,bk);
            % Project whole spatial patch through leadfield
            smthlead = UL * q;
            
            % Normalize prior weighting
            den = smthlead' * smthlead;
            if ~(isfinite(den) && den > tiny)
                continue;
            end
            num = smthlead' * InvCov * smthlead;
            if ~(isfinite(num) && num > tiny)
                continue;
            end
            normpower = 1 / den;
            srcpow    = 1 / num;
            val = srcpow / normpower;
            if isfinite(val) && val > 0
                ind_source(bk) = val;
            end
        end

        % 2) Enumerate all possible laminar pairs (55 for 11 layers)
        nPairs = nlayers * (nlayers - 1) / 2;
        layer_pairs = zeros(nPairs,2);
        c = 0;
        for a = 1:nlayers-1
            for b = a+1:nlayers
                c = c + 1;
                layer_pairs(c,:) = [a b];
            end
        end

        % 3) Calculate evidence for each SUM and DIFFERENCE pair
        %
        % SUM:
        %   Optimize the original raw EBB score continuously over the
        %   within-pair mixing weight r:
        %
        %       q+(r) = qa + r*qb,  r >= 0.
        %
        %   For each anatomical pair and cortical column, the score is
        %
        %       S(r) = (q'L'Lq) / (4 q'L'InvCov Lq).
        %
        %   Because this is a two-dimensional generalized Rayleigh quotient,
        %   the stationary condition is quadratic in r. We evaluate all
        %   positive real stationary roots plus the two endpoints r=0 and
        %   r=Inf.
        %
        %   Endpoint optima are retained as diagnostics in pair_sum_raw but
        %   are NOT admitted to the SUM family: they are single-layer
        %   hypotheses already represented by IND. pair_sum therefore
        %   contains only genuine interior SUM evidence.
        %
        % DIFF:
        %   Retains the original fixed q- = qa - qb hypothesis.
        %
        % Dimensions:
        %
        %   cortical column x layer-pair
        pair_sum          = zeros(V, nPairs);
        pair_sum_raw      = zeros(V, nPairs);
        pair_sum_mixing    = NaN(V, nPairs);
        pair_sum_endpoint = true(V, nPairs);
        pair_sum_interior = false(V, nPairs);
        pair_diff         = zeros(V, nPairs);

        % Numerical tolerance used only to decide whether the quadratic
        % stationary equation has effectively lost its leading term.
        sum_root_tol = 1e-12;

        % For each cortical column, first project the 11 layer patches once.
        % This gives local 11 x 11 numerator/denominator quadratic forms,
        % avoiding repeated lead-field projection for every candidate r.
        for bk = 1:V

            idx_layers = bk + (0:nlayers-1)*V;
            layer_leads = UL * QG(:,idx_layers);

            G_local = layer_leads' * layer_leads;
            N_local = layer_leads' * InvCov * layer_leads;

            % Remove negligible numerical asymmetry.
            G_local = (G_local + G_local') / 2;
            N_local = (N_local + N_local') / 2;

            % For each layer pair
            for p = 1:nPairs

                la = layer_pairs(p,1);
                lb = layer_pairs(p,2);

                gaa = G_local(la,la);
                gab = G_local(la,lb);
                gbb = G_local(lb,lb);

                naa = N_local(la,la);
                nab = N_local(la,lb);
                nbb = N_local(lb,lb);

                % ---------------------------------------------------------
                % Continuous SUM optimum
                %
                % For c(r)=[1;r],
                %
                %   S(r) = (gaa + 2*r*gab + r^2*gbb) /
                %          (4*(naa + 2*r*nab + r^2*nbb)).
                %
                % dS/dr = 0 reduces to
                %
                %   A2*r^2 + A1*r + A0 = 0
                %
                % with:
                %
                %   A2 = gbb*nab - gab*nbb
                %   A1 = gbb*naa - gaa*nbb
                %   A0 = gab*naa - gaa*nab
                % ---------------------------------------------------------
                best_sum_val      = 0;
                best_sum_mixing    = NaN;
                best_sum_endpoint = true;

                % Endpoint A: r = 0
                if isfinite(gaa) && isfinite(naa) && ...
                        gaa > tiny && naa > tiny
                    val = gaa / (4 * naa);

                    if isfinite(val) && val > best_sum_val
                        best_sum_val      = val;
                        best_sum_mixing    = 0;
                        best_sum_endpoint = true;
                    end
                end

                A2 = gbb*nab - gab*nbb;
                A1 = gbb*naa - gaa*nbb;
                A0 = gab*naa - gaa*nab;

                coeff_scale = max(abs([A2 A1 A0]));
                positive_roots = [];

                if isfinite(coeff_scale) && coeff_scale > 0
                    coeff_tol = sum_root_tol * coeff_scale;

                    if abs(A2) > coeff_tol
                        disc = A1*A1 - 4*A2*A0;

                        if isfinite(disc) && disc >= 0
                            root_disc = sqrt(disc);
                            r1 = (-A1 + root_disc) / (2*A2);
                            r2 = (-A1 - root_disc) / (2*A2);

                            if isfinite(r1) && r1 > 0
                                positive_roots(end+1) = r1; %#ok<AGROW>
                            end

                            if isfinite(r2) && r2 > 0
                                positive_roots(end+1) = r2; %#ok<AGROW>
                            end
                        end

                    elseif abs(A1) > coeff_tol
                        r1 = -A0 / A1;

                        if isfinite(r1) && r1 > 0
                            positive_roots(end+1) = r1; %#ok<AGROW>
                        end
                    end
                end

                % Positive interior stationary points.
                for ridx = 1:numel(positive_roots)
                    r = positive_roots(ridx);

                    den = gaa + 2*r*gab + (r^2)*gbb;
                    num = naa + 2*r*nab + (r^2)*nbb;

                    if isfinite(den) && den > tiny && ...
                            isfinite(num) && num > tiny
                        val = den / (4 * num);

                        if isfinite(val) && val > best_sum_val
                            best_sum_val      = val;
                            best_sum_mixing    = r;
                            best_sum_endpoint = false;
                        end
                    end
                end

                % Endpoint B: r = Inf
                if isfinite(gbb) && isfinite(nbb) && ...
                        gbb > tiny && nbb > tiny
                    val = gbb / (4 * nbb);

                    if isfinite(val) && val > best_sum_val
                        best_sum_val      = val;
                        best_sum_mixing    = Inf;
                        best_sum_endpoint = true;
                    end
                end

                pair_sum_raw(bk,p)      = best_sum_val;
                pair_sum_mixing(bk,p)    = best_sum_mixing;
                pair_sum_endpoint(bk,p) = best_sum_endpoint;

                % Only genuine two-layer interior optima enter SUM.
                if ~best_sum_endpoint && ...
                        isfinite(best_sum_mixing) && ...
                        best_sum_mixing > 0 && ...
                        isfinite(best_sum_val) && ...
                        best_sum_val > 0

                    pair_sum(bk,p)          = best_sum_val;
                    pair_sum_interior(bk,p) = true;
                end

                % ---------------------------------------------------------
                % DIFFERENCE hypothesis
                %
                % q- = qa - qb
                %
                % Calculated from the same local quadratic forms.
                % ---------------------------------------------------------
                den = gaa + gbb - 2*gab;
                num = naa + nbb - 2*nab;

                if isfinite(den) && den > tiny && ...
                        isfinite(num) && num > tiny
                    val = den / (4 * num);

                    if isfinite(val) && val > 0
                        pair_diff(bk,p) = val;
                    end
                end
            end
        end

        fprintf( ...
            ['EBBlayer SUM optimisation: continuous interior; ' ...
             'endpoint optima excluded\n'] ...
        );

        fprintf( ...
            ['EBBlayer SUM endpoint optima: %d / %d ' ...
             'vertex-pair hypotheses (%.2f%%)\n'], ...
            nnz(pair_sum_endpoint), ...
            numel(pair_sum_endpoint), ...
            100 * nnz(pair_sum_endpoint) / numel(pair_sum_endpoint) ...
        );

        % =========================================================================
        % Select TOP-K SUM hypotheses independently at each cortical column
        %
        % pair_sum already contains only genuine interior SUM optima.
        % Endpoint solutions are zero and therefore cannot be selected.
        % =========================================================================
        if EBBlayer_sum_pair_topk > nPairs
            error( ...
                ['EBBlayer_sum_pair_topk=%d exceeds the number of possible ' ...
                 'layer pairs (%d).'], ...
                EBBlayer_sum_pair_topk, ...
                nPairs ...
            );
        end

        keep_sum = false(V, nPairs);
        for bk = 1:V
            scores = pair_sum(bk, :);
            scores(~isfinite(scores)) = 0;

            positive_idx = find(scores > 0);
            if isempty(positive_idx)
                continue;
            end

            [~, local_order] = sort(scores(positive_idx), 'descend');
            n_keep = min(EBBlayer_sum_pair_topk, length(positive_idx));
            chosen = positive_idx(local_order(1:n_keep));
            keep_sum(bk, chosen) = true;
        end

        % Defensive enforcement: only genuine interior SUM hypotheses can enter
        % the covariance, even if the scoring code is changed later.
        keep_sum = keep_sum & pair_sum_interior & (pair_sum > 0);

        n_sum_pairs_kept = sum(any(keep_sum, 1));
        n_sum_pair_entries_kept = nnz(keep_sum);

        fprintf( ...
            ['EBBlayer SUM TOP-K: K=%d, %d pair families active somewhere, ' ...
             '%d vertex-pair entries retained\n'], ...
            EBBlayer_sum_pair_topk, ...
            n_sum_pairs_kept, ...
            n_sum_pair_entries_kept ...
        );

        % =========================================================================
        % Select TOP-K DIFF hypotheses independently at each cortical column
        % =========================================================================
        if EBBlayer_diff_pair_topk > nPairs
            error( ...
                ['EBBlayer_diff_pair_topk=%d exceeds the number of possible ' ...
                 'layer pairs (%d).'], ...
                EBBlayer_diff_pair_topk, ...
                nPairs ...
            );
        end

        keep_diff = false(V, nPairs);
        for bk = 1:V
            scores = pair_diff(bk, :);
            scores(~isfinite(scores)) = 0;

            positive_idx = find(scores > 0);
            if isempty(positive_idx)
                continue;
            end

            [~, local_order] = sort(scores(positive_idx), 'descend');
            n_keep = min(EBBlayer_diff_pair_topk, length(positive_idx));
            chosen = positive_idx(local_order(1:n_keep));
            keep_diff(bk, chosen) = true;
        end

        n_diff_pairs_kept = sum(any(keep_diff, 1));
        n_diff_pair_entries_kept = nnz(keep_diff);

        fprintf( ...
            ['EBBlayer DIFF TOP-K: K=%d, %d pair families active somewhere, ' ...
             '%d vertex-pair entries retained\n'], ...
            EBBlayer_diff_pair_topk, ...
            n_diff_pairs_kept, ...
            n_diff_pair_entries_kept ...
        );

        % =========================================================================
        % Save pairwise evidence at one diagnostic cortical column
        % =========================================================================
        if ~isempty(EBBlayer_diag_vertex)
            if EBBlayer_diag_vertex < 1 || EBBlayer_diag_vertex > V
                error( ...
                    ['EBBlayer_diag_vertex=%d is outside valid range ' ...
                     '1..%d.'], ...
                    EBBlayer_diag_vertex, ...
                    V ...
                );
            end
            pair_sum_diag = pair_sum(EBBlayer_diag_vertex, :)';
            pair_sum_raw_diag = pair_sum_raw(EBBlayer_diag_vertex, :)';
            pair_sum_mixing_diag = pair_sum_mixing(EBBlayer_diag_vertex, :)';
            pair_sum_endpoint_diag = pair_sum_endpoint(EBBlayer_diag_vertex, :)';
            pair_sum_interior_diag = pair_sum_interior(EBBlayer_diag_vertex, :)';
            pair_keep_diag = keep_sum(EBBlayer_diag_vertex, :)';
            pair_sum_used_diag = pair_sum_diag .* double(pair_keep_diag);
            pair_sum_mixing_used_diag = pair_sum_mixing_diag;
            pair_sum_mixing_used_diag(~pair_keep_diag) = NaN;
            pair_diff_diag = pair_diff(EBBlayer_diag_vertex, :)';
            pair_keep_diff_diag = keep_diff(EBBlayer_diag_vertex, :)';
            pair_diff_used_diag = pair_diff_diag .* double(pair_keep_diff_diag);

            % Rank post-endpoint-exclusion SUM scores at this cortical column.
            pair_sum_rank_diag = NaN(nPairs,1);
            eligible_sum = find(isfinite(pair_sum_diag) & pair_sum_diag > 0);
            for pp = eligible_sum(:)'
                pair_sum_rank_diag(pp) = ...
                    1 + sum(pair_sum_diag > pair_sum_diag(pp));
            end

            % DIFF ranks are useful for checking SUM/DIFF union coverage.
            pair_diff_rank_diag = NaN(nPairs,1);
            eligible_diff = find(isfinite(pair_diff_diag) & pair_diff_diag > 0);
            for pp = eligible_diff(:)'
                pair_diff_rank_diag(pp) = ...
                    1 + sum(pair_diff_diag > pair_diff_diag(pp));
            end

            EBBlayer_diag_out.pair_sum_used_vertex = pair_sum_used_diag;
            EBBlayer_diag_out.pair_sum_raw_vertex = pair_sum_raw_diag;
            EBBlayer_diag_out.pair_sum_mixing_vertex = pair_sum_mixing_diag;
            EBBlayer_diag_out.pair_sum_endpoint_vertex = pair_sum_endpoint_diag;
            EBBlayer_diag_out.pair_sum_interior_vertex = pair_sum_interior_diag;
            EBBlayer_diag_out.pair_sum_rank_vertex = pair_sum_rank_diag;
            EBBlayer_diag_out.pair_diff_rank_vertex = pair_diff_rank_diag;
            EBBlayer_diag_out.pair_sum_mixing_used_vertex = ...
                pair_sum_mixing_used_diag;
            % Explicit SUM and DIFF selection masks.
            EBBlayer_diag_out.pair_keep_sum_vertex = pair_keep_diag;
            EBBlayer_diag_out.pair_keep_diff_vertex = pair_keep_diff_diag;
            EBBlayer_diag_out.pair_diff_used_vertex = pair_diff_used_diag;
            EBBlayer_diag_out.diag_vertex = EBBlayer_diag_vertex;
            
            mx_diff_used = max(pair_diff_used_diag);
            if isfinite(mx_diff_used) && mx_diff_used > 0
                EBBlayer_diag_out.pair_diff_used_vertex_norm = ...
                    pair_diff_used_diag / mx_diff_used;
            else
                EBBlayer_diag_out.pair_diff_used_vertex_norm = ...
                    pair_diff_used_diag;
            end

            EBBlayer_diag_out.layer_pairs = layer_pairs;           
            EBBlayer_diag_out.pair_sum_vertex = pair_sum_diag;
            EBBlayer_diag_out.pair_diff_vertex = pair_diff_diag;

            mx_sum_raw = max(pair_sum_raw_diag);
            if isfinite(mx_sum_raw) && mx_sum_raw > 0
                EBBlayer_diag_out.pair_sum_raw_vertex_norm = ...
                    pair_sum_raw_diag / mx_sum_raw;
            else
                EBBlayer_diag_out.pair_sum_raw_vertex_norm = ...
                    pair_sum_raw_diag;
            end

            % Within-row normalization for ranking/visualization only.
            % pair_sum_vertex is the post-endpoint-exclusion score actually
            % eligible for SUM covariance construction.
            mx_sum = max(pair_sum_diag);
            if isfinite(mx_sum) && mx_sum > 0
                EBBlayer_diag_out.pair_sum_vertex_norm = ...
                    pair_sum_diag / mx_sum;
            else
                EBBlayer_diag_out.pair_sum_vertex_norm = ...
                    pair_sum_diag;
            end

            mx_diff = max(pair_diff_diag);
            if isfinite(mx_diff) && mx_diff > 0
                EBBlayer_diag_out.pair_diff_vertex_norm = ...
                    pair_diff_diag / mx_diff;
            else
                EBBlayer_diag_out.pair_diff_vertex_norm = ...
                    pair_diff_diag;
            end
        end

        % 4) Construct cross-layer covariance matrices
        %
        % For each layer pair a,b:
        % SUM:
        %       weighted rank-1 block using the winning pair-specific mixing weight r
        %
        %       v * c(r)c(r)'
        %
        % where
        %
        %       c(r) = sqrt(2/(1+r^2)) * [1; r]
        %
        % so every mixing weight has the same basis-vector norm, and r=1 reduces
        % exactly to the original v * [1;1] * [1 1] block.
        %
        % DIFFERENCE:
        %       [ +v  -v ]
        %       [ -v  +v ]
        % = v * [1;-1] * [1 -1]
        %
        % Maximum possible number of non-zero elements:
        % each cortical column has an nlayers x nlayers block in the matrix
        nz_est = V * nlayers * nlayers;

        % Initialize sparse matrices
        Qsum  = spalloc(Ns, Ns, nz_est);
        Qdiff = spalloc(Ns, Ns, nz_est);

        % Construct diagonal terms separately.
        % V x nlayers
        diag_sum  = zeros(V, nlayers);
        diag_diff = zeros(V, nlayers);

        % =========================================================================
        % Construct SUM / DIFF covariance matrices from pair scores
        %
        % SUM:
        %
        %   pair_sum contains the continuous interior SUM evidence.
        %   keep_sum(:,p) determines which SUM pairs are retained.
        %
        % DIFF:
        %
        %   pair_diff contains the raw DIFF pair evidence.
        %   keep_diff(:,p) independently determines which DIFF pairs are retained.
        %
        % SUM and DIFF pair selection are independent.
        % =========================================================================

        % For each layer pair
        for p = 1:nPairs

            la = layer_pairs(p,1);
            lb = layer_pairs(p,2);

            % ---------------------------------------------------------------------
            % Indices in the full multilayer covariance matrices
            % ---------------------------------------------------------------------
            ia = (la-1)*V + (1:V);
            ib = (lb-1)*V + (1:V);

            % ---------------------------------------------------------------------
            % SUM weights
            %
            % Use the continuous interior SUM score, then apply the
            % vertex-specific TOP-K inclusion mask.
            % ---------------------------------------------------------------------
            vs = pair_sum(:, p);
            vs = vs .* double(keep_sum(:, p));

            % ---------------------------------------------------------------------
            % DIFFERENCE weights
            %
            % Use raw DIFF pair evidence, then independently apply the
            % vertex-specific DIFF inclusion mask.
            % ---------------------------------------------------------------------
            vd = pair_diff(:, p);
            vd = vd .* double(keep_diff(:, p));

            % =====================================================================
            % SUM covariance
            %
            % For each cortical column, use the mixing weight that maximised the raw
            % SUM score for this anatomical pair.
            %
            % Normalise the weighted direction so ||c(r)||^2 = 2:
            %
            %   c(r) = sqrt(2/(1+r^2)) * [1; r]
            %
            % This keeps pair-basis scale constant across mixing weights and preserves
            % the original SUM block exactly at r=1.
            % =====================================================================
            if any(keep_sum(:, p))
                % IMPORTANT:
                % pair_sum_mixing contains 0 / Inf for endpoint optima that have
                % been excluded from SUM.  Do not evaluate the mixing-weight formulas
                % over those rows and then multiply by zero: expressions such
                % as 0 * Inf produce NaN and contaminate Qsum.
                %
                % Work only on genuinely retained interior rows.
                kept_idx = find(keep_sum(:, p));
                rs = pair_sum_mixing(kept_idx, p);
                vk = vs(kept_idx);

                if any(~isfinite(rs)) || any(rs <= 0)
                    error('Retained SUM pair has invalid interior mixing weight.');
                end

                scale2_kept = 2 ./ (1 + rs.^2);

                sum_aa  = zeros(V,1);
                sum_bb  = zeros(V,1);
                sum_off = zeros(V,1);

                sum_aa(kept_idx) = vk .* scale2_kept;
                sum_bb(kept_idx) = vk .* scale2_kept .* (rs.^2);
                sum_off(kept_idx) = vk .* scale2_kept .* rs;

                Ws_off = spdiags(sum_off, 0, V, V);

                % Off-diagonal blocks
                Qsum(ia, ib) = Ws_off;
                Qsum(ib, ia) = Ws_off;

                % Diagonal contribution
                diag_sum(:, la) = diag_sum(:, la) + sum_aa;
                diag_sum(:, lb) = diag_sum(:, lb) + sum_bb;
            end

            % =====================================================================
            % DIFFERENCE covariance
            % =====================================================================
            if any(keep_diff(:, p))
                Wd = spdiags(vd, 0, V, V);

                % Negative off-diagonal covariance
                Qdiff(ia, ib) = -Wd;
                Qdiff(ib, ia) = -Wd;

                % Positive diagonal contributions
                diag_diff(:, la) = diag_diff(:, la) + vd;
                diag_diff(:, lb) = diag_diff(:, lb) + vd;
            end
        end

        % Add diagonal blocks.
        % DIFF retains the original [v -v; -v v] pair block.
        % SUM diagonal terms were accumulated above using the selected
        % mixing-weight-specific, norm-matched rank-1 block.
        for l = 1:nlayers
            idx = (l-1)*V + (1:V);
            Qsum(idx,idx) = spdiags(diag_sum(:,l), 0, V, V);
            Qdiff(idx,idx) = spdiags(diag_diff(:,l), 0, V, V);
        end

        % Force matrices to be symmetric
        Qsum  = (Qsum  + Qsum')  / 2;
        Qdiff = (Qdiff + Qdiff') / 2;

        % 5) Clean and normalise the three prior components
        % Normalize independent source family
        ind_source(~isfinite(ind_source)) = 0;
        mx = full(max(ind_source));
        if mx > 0 && isfinite(mx)
            ind_source = ind_source / mx;
        end

        % Normalise covariance families by maximum diagonal variance
        d = full(diag(Qsum));
        d(~isfinite(d)) = 0;
        mx = max(d);
        if mx > 0 && isfinite(mx)
            Qsum = Qsum / mx;
        end

        d = full(diag(Qdiff));
        d(~isfinite(d)) = 0;
        mx = max(d);
        if mx > 0 && isfinite(mx)
            Qdiff = Qdiff / mx;
        end

        % 6) Supply three source covariance components to ReML
        % source prior covariance for each family
        Qp   = {};
        Qp{1} = spdiags(ind_source,0,Ns,Ns);
        Qp{2} = Qsum;
        Qp{3} = Qdiff;

        % source prior covariance Qp, for each family, projected through 
        % the lead field into sensor/spatial-mode space and normalized
        LQpL = cell(1,3);
        for i = 1:3
            LQpL{i} = UL * Qp{i} * UL';
            tr = full(trace(LQpL{i}));
            if isfinite(tr) && tr > 0
                Qp{i}   = Qp{i}   / tr;
                LQpL{i} = LQpL{i} / tr;
            end
        end

        % =========================================================================
        % Save the final trace-normalized local covariance blocks
        % =========================================================================
        if strcmp(type, 'EBBlayer') && ~isempty(EBBlayer_diag_vertex)
            diag_idx = EBBlayer_diag_vertex + (0:nlayers-1) * V;
            EBBlayer_diag_out.Qind_local = full(Qp{1}(diag_idx,diag_idx));
            EBBlayer_diag_out.Qsum_local = full(Qp{2}(diag_idx,diag_idx));
            EBBlayer_diag_out.Qdiff_local = full(Qp{3}(diag_idx,diag_idx));
        end
        
        % Diagnostics
        tr1 = full(sum(spdiags(LQpL{1},0)));
        tr2 = full(sum(spdiags(LQpL{2},0)));
        tr3 = full(sum(spdiags(LQpL{3},0)));

        fprintf('trace(LQpL): ind=%.3g, sum=%.3g, diff=%.3g\n', tr1, tr2, tr3);
        fprintf('nnz(Qp): ind=%d, sum=%d, diff=%d\n', nnz(Qp{1}), nnz(Qp{2}), nnz(Qp{3}));
        
        if strcmp(type, 'EBBlayer')
            % Algorithm provenance and tunable TOP-K parameters.
            EBBlayer_diag_out.sum_pair_topk = EBBlayer_sum_pair_topk;
            EBBlayer_diag_out.diff_pair_topk = EBBlayer_diff_pair_topk;
            EBBlayer_diag_out.sum_pair_optimization = 'continuous_interior';
            EBBlayer_diag_out.sum_endpoint_exclusion = true;
            EBBlayer_diag_out.sum_root_tolerance = sum_root_tol;

            EBBlayer_diag_out.n_sum_endpoint_entries = nnz(pair_sum_endpoint);
            EBBlayer_diag_out.n_sum_interior_entries = nnz(pair_sum_interior);
            EBBlayer_diag_out.n_sum_pairs_kept = n_sum_pairs_kept;
            EBBlayer_diag_out.n_sum_pair_entries_kept = n_sum_pair_entries_kept;
            EBBlayer_diag_out.n_vertices_with_full_sum_topk = ...
                sum(sum(keep_sum, 2) == EBBlayer_sum_pair_topk);

            EBBlayer_diag_out.n_diff_pairs_kept = n_diff_pairs_kept;
            EBBlayer_diag_out.n_diff_pair_entries_kept = n_diff_pair_entries_kept;
            EBBlayer_diag_out.n_vertices_with_full_diff_topk = ...
                sum(sum(keep_diff, 2) == EBBlayer_diff_pair_topk);
        end

        % Summary
        % Qp      = source-space covariance hypothesis
        % UL      = source to sensor/spatial-mode mapping
        % LQpL    = predicted sensor covariance from that hypothesis
        % AYYA    = observed sensor covariance
        % ReML    = finds the mixture of LQpL components that best explains AYYA
        
    case {'EBBgs'}
        allsource = zeros(Ntrials,Ns);
        for ii = 1:Ntrials
            InvCov = spm_inv(YYep{ii});
            Sourcepower = zeros(Ns,1);
            for bk = 1:Ns
                normpower = 1/(UL(:,bk)'*UL(:,bk));
                Sourcepower(bk) = 1/(UL(:,bk)'*InvCov*UL(:,bk));
                allsource(ii,bk) = Sourcepower(bk)./normpower;
            end
            
            Qp{ii}.q = allsource(ii,:);
        end
        
    case {'LOR','COH'}
        Qp{1} = speye(Ns,Ns);
        LQpL{1} = UL*UL';
        
        Qp{2} = QG;
        LQpL{2} = UL*Qp{2}*UL';
        
    case {'IID','MMN'}
        Qp{1} = speye(Ns,Ns);
        LQpL{1} = UL*UL';
end

fprintf('Using %d spatial source priors provided\n',length(Qp));

QP = {};
LQP = {};
LQPL = {};

switch(type)
    case {'MSP','GS','EBBgs'}
        Np = length(Qp);
        Q = zeros(Ns,Np);
        for i = 1:Np
            Q(:,i) = Qp{i}.q;
        end
        Q = sparse(Q);
        
        MVB = spm_mvb(AY,UL,[],Q,Qe,16);
        
        Qcp = Q*MVB.cp;
        QP{end + 1} = sum(Qcp.*Q,2);
        LQP{end + 1} = (UL*Qcp)*Q';
        LQPL{end + 1} = LQP{end}*UL';
end

switch(type)
    case {'MSP','ARD'}
        [Cy,h,Ph,F_out] = spm_sp_reml(AYYA,[],[Qe LQpL],Nn);
        
        Ne = length(Qe);
        Np = length(Qp);
        
        hp = h(Ne + (1:Np));
        
        qp = sparse(0);
        for i = 1:Np
            if hp(i) > max(hp)/128
                qp = qp + hp(i)*Qp{i}.q*Qp{i}.q';
            end
        end
        
        QP{end + 1} = diag(qp);
        LQP{end + 1} = UL*qp;
        LQPL{end + 1} = LQP{end}*UL';
end

switch(type)
    case {'IID','MMN','LOR','COH','EBB','EBBcorr','EBBlayer'}
        [Cy,h,Ph,F_out] = spm_reml_sc( ...
            AYYA, [], [Qe LQpL], Nn, -4, 16, Q0);

        Ne = length(Qe);
        Np = length(Qp);

        % These are the source-family ReML weights we actually care about
        source_hp = full(h(Ne + (1:Np)));
        source_hp = source_hp(:);


        % -------------------------------------------------------------------------
        % Save EBBlayer source-family diagnostics BEFORE the second ReML stage
        % -------------------------------------------------------------------------
        if strcmp(type, 'EBBlayer')
            EBBlayer_diag_out.labels = {
                'IND'
                'SUM'
                'DIFF'
            };
            EBBlayer_diag_out.hp_source = source_hp;
            EBBlayer_diag_out.source_reml_F = full(F_out);

            % -------------------------------------------------------------
            % Save normalized sensor-space covariance components.
            % These tell us whether the COMPONENT SHAPES themselves
            % change as rho changes, independently of their ReML weights.
            % -------------------------------------------------------------
            EBBlayer_diag_out.LQpL = cell(1,3);
            component_trace = zeros(3,1);
            for i = 1:3
                EBBlayer_diag_out.LQpL{i} = full(LQpL{i});
                component_trace(i) = full(trace(LQpL{i}));
            end
            EBBlayer_diag_out.component_trace = component_trace;

            % Because trace-normalized, this should essentially equal source_hp.
            EBBlayer_diag_out.effective_weight = source_hp .* component_trace;

            fprintf('hp: ind=%.3g, sum=%.3g, diff=%.3g\n', source_hp(1), source_hp(2), source_hp(3));
            fprintf('trace: ind=%.3g, sum=%.3g, diff=%.3g\n', component_trace(1), component_trace(2), component_trace(3));
        end

        % =========================================================================
        % Construct the EBBlayer source prior from the ReML-estimated family
        % weights:
        %
        %     qp = h_IND  * Q_IND
        %        + h_SUM  * Q_SUM
        %        + h_DIFF * Q_DIFF
        %
        % Qp{1}, Qp{2}, and Qp{3} have already been independently normalized
        % so that trace(UL * Qp{i} * UL') = 1.
        % =========================================================================
        qp = sparse(0);
        for i = 1:Np
            qp = qp + source_hp(i) * Qp{i};
        end

        % =========================================================================
        % Convert the chosen full source covariance into the representation used by
        % the final ReML stage.
        %
        % IMPORTANT:
        %
        % LQP retains the FULL covariance:
        %
        %     UL * qp
        %
        % whereas QP contains only diag(qp) because it is used downstream for the
        % posterior marginal variance Cq.
        % =========================================================================
        QP{end + 1} = diag(qp);
        LQP{end + 1} = UL * qp;
        LQPL{end + 1} = LQP{end} * UL';
end

fprintf('Inverting subject 1\n')

Np = length(LQPL);
Ne = length(Qe);

Q = [{Q0} LQPL];

if rank(AYYA)~=size(A,1)
    rank(AYYA);
    size(AYYA,1);
    warning('AYYA IS RANK DEFICIENT');
end

[Cy,h,Ph,F_out] = spm_reml_sc(AYYA, [], Q, Nn, -4, 16, Q0);

Cp = sparse(0);
LCp = sparse(0);

final_hp = full(h(Ne + (1:Np)));
final_hp = final_hp(:);


if strcmp(type, 'EBBlayer')
    EBBlayer_diag_out.hp_final_scale = final_hp;
    EBBlayer_diag_out.final_reml_F = full(F_out);
end

for j = 1:Np
    Cp = Cp + final_hp(j) * QP{j};
    LCp = LCp + final_hp(j) * LQP{j};
end

M = LCp'/Cy;

Cq = Cp - sum(LCp.*M')';

SSR = 0;
SST = 0;
J = {};

for j = 1:Ntrialtypes
    J{j} = M*UY{j};
    
    SSR = SSR + sum(var((UY{j} - UL*J{j})));
    SST = SST + sum(var(UY{j}));
end

R2_out = 100*(SST - SSR)/SST;
fprintf('Percent variance explained %.2f (%.2f)\n',full(R2_out),full(R2_out*VE_out));

% Outputs for combining later
J_out = J;
M_out = M;
Cq_out = Cq;
U_out = U;
V_out = V;
Vq_out = Vq;
S_out = S;
It_out = It;
Ik_out = Ik;
ID_out = ID;
pst_out = pst;
dct_out = dct;
end