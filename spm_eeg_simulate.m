function [Dnew,meshsourceind]=spm_eeg_simulate(D,prefix,patchmni,simsignal,ormni,woi,whitenoise,SNRdB,trialind,mnimesh,dipfwhm,nAmdipmom);
% Simulate a number of MSP patches at specified locations on existing mesh
% FORMAT [Dnew,meshsourceind]=spm_eeg_simulate(D,prefix,patchmni,simsignal,woi,whitenoise,SNRdB,trialind,mnimesh,dipfwhm);
% D dataset or dataset filename
% prefix : prefix of new simulated dataset
% patchmni : patch centres in mni space or patch indices
% simsignal : Nsources x time series in nAm within woi
% woi: window of interest in seconds
% whitenoise level in rms femto Tesla or micro volts
% SNRdB power signal to noise ratio in dBs
% trialind: trials on which the simulated data will be added to the noise
% mnimesh : a new mesh with vertices in mni space
% dipfwhm - patch smoothness in mm
%
% Outputs
% Dnew- new dataset
% meshsourceind- vertex indices of sources on the mesh
%__________________________________________________________________________

% Jose David Lopez, Gareth Barnes, Vladimir Litvak
% Copyright (C) 2013-2022 Wellcome Centre for Human Neuroimaging


% LOAD IN ORIGINAL DATA
%==========================================================================
useind=1; % D to use
if ischar(D) || isstring(D)
    data=spm_eeg_load(D);
    D={data};
end
if nargin<2,
    prefix='';
end;

if nargin<3,
    patchmni=[];
end;


if nargin<4,
    simsignal=[];
end;
if nargin<5,
    ormni=[];
end;

if nargin<6,
    woi=[];
end;


if nargin<7,
    whitenoise=[];
end;

if nargin<8,
    SNRdB=[];
end;

if nargin<9,
    trialind=[];
end;

if nargin<10,
    mnimesh=[];
end;



if nargin<11
    dipfwhm=[]; %% number of iterations used to smooth patch out (more iterations, larger patch)
end;



if isempty(prefix),
    prefix='sim';
end;

if isempty(dipfwhm),
    dipfwhm=6; %% FWHM in mm
end;

if isempty(woi),
    woi=[D{useind}.time(1) D{useind}.time(end)];
end;


val=D{useind}.val;



if isempty(patchmni),
    patchmni=[-45.4989  -30.6967    4.9213;...
        46.7322  -31.2311    4.0085];
    
end;


if ~xor(isempty(whitenoise),isempty(SNRdB))
    error('Must specify either white noise level or sensor level SNR');
end;





[a1 b1 c1]=fileparts(D{useind}.fname);
newfilename=[prefix b1 c1];

%% forcing overwrite of an existing file
Dnew=D{useind}.clone(newfilename);


if isempty(trialind)
    trialind=1:Dnew.ntrials;
end;

modstr=deblank(modality(D{1}));
disp(sprintf('Simulating data on %s channels only',modstr));



if ~isempty(mnimesh),
    Dnew.inv{val}.mesh.tess_mni.vert=mnimesh.vert;
    Dnew.inv{val}.mesh.tess_mni.face=mnimesh.face;
    Dnew.inv{val}.forward.mesh.vert=spm_eeg_inv_transform_points(Dnew.inv{val}.datareg.fromMNI,mnimesh.vert);
    Dnew.inv{val}.forward.mesh.face=mnimesh.face;
end; % if

% Two synchronous sources
if ~isempty(patchmni),
    Ndips=size(patchmni,1);
else
    Ndips=0;
end;

if size(simsignal,1)~=Ndips,
    error('number of signals given does not match number of sources');
end;

meshsourceind=[];

disp('Using closest mesh vertices to the specified coordinates')
for d=1:Ndips,
    vdist= Dnew.inv{val}.mesh.tess_mni.vert-repmat(patchmni(d,:),size(Dnew.inv{val}.mesh.tess_mni.vert,1),1);
    dist=sqrt(dot(vdist',vdist'));
    [mnidist(d),meshsourceind(d)] =min(dist);
end;

disp(sprintf('Furthest distance from dipole location to mesh %3.2f mm',max(mnidist)));
if max(mnidist)>0.1
    warning('Supplied vertices do not sit on the mesh!');
end;


Ndip = size(simsignal,1);       % Number of dipoles



sensorunits = Dnew.units; %% of sensors (T or fT)

try Dnew.inv{val}.forward.vol.unit, %% units of forward model for distance (m or mm)
    switch(Dnew.inv{val}.forward.vol.unit), %% correct for non-SI lead field scaling
        case 'mm'
            
            Lscale=1000*1000;
        case 'cm'
            
            Lscale=100*100;
        case 'm'
            Lscale=1.0;
            
        otherwise
            error('unknown volume unit');
            
    end;
catch
    disp('No distance units found');
    Lscale=1.0;
end;





Ntrials = Dnew.ntrials;             % Number of trials

% define period over which dipoles are active
startf1  = woi(1);                  % (sec) start time
endf1 = woi(2); %% end time
f1ind = intersect(find(Dnew.time>startf1),find(Dnew.time<=endf1));

if length(f1ind)~=size(simsignal,2),
    error('Signal does not fit in time window');
end;


%if isequal(modstr, 'MEG')
try
    chanind = Dnew.indchantype({'MEG', 'MEGPLANAR'}, 'GOOD');
catch
    chanind = Dnew.indchantype(modality(D{1}), 'GOOD');
end






labels=Dnew.chanlabels(chanind);


%chans = Dnew.indchantype(modstr, 'GOOD');

simscale=1.0;

try
    %% white noise is input in fT or uV so convert it to data sensorunits
    switch sensorunits{chanind(1)}
        case 'T'
            simscale=1e-15; %% convert from fT to T
            %whitenoise=whitenoise./1e15; %% rms femto Tesla
            %tmp=tmp./1e15; %% also computed in fT originally
        case 'fT'
            simscale=1.0; %% sensors already in fT
            %whitenoise=whitenoise; %% rms  Tesla
            %tmp=tmp;
        case 'uV'
            whitenoise=whitenoise; %% micro volts
            tmp=tmp;
        case 'V'
            whitenoise=whitenoise./1e6; %% volts
            tmp=tmp./1e6;
            error('not supported for EEG at the moment');
            
        otherwise
            error('unknown sensor unit')
    end;
    
catch
    disp('No sensor sensorunits found');
end;

whitenoise=whitenoise.*simscale;



if ~isempty(ormni), %%%% DIPOLE SIMULATION
    disp('SIMULATING DIPOLE SOURCES');
    
    Nchans=length(chanind);
    
    if size(ormni)~=size(patchmni),
        error('A 3D orientation must be specified for each source location');
    end;
    
    posdipmm=Dnew.inv{val}.datareg.fromMNI*[patchmni ones(size(ormni,1),1)]'; %% put into MEG space
    posdipmm=posdipmm(1:3,:)';
    %% need to make a pure rotation for orientation transform to native space
    M=Dnew.inv{val}.datareg.fromMNI*Dnew.inv{val}.mesh.Affine;            
    ordip=[ormni ones(size(ormni,1),1)]*inv(M')';
    ordip=ordip(:,1:3);
            
    for i=1:size(ordip,1)
        ordip(i,:)=ordip(i,:)./sqrt(dot(ordip(i,:),ordip(i,:))); %% make sure it is unit vector
    end
    
    %% NB COULD ADD A PURE DIPOLE SIMULATION IN FUTURE
    sens=Dnew.inv{val}.forward.sensors;
    vol=Dnew.inv{val}.forward.vol;
    
    sensorind=Dnew.indchantype({'MEG', 'MEGPLANAR','REFMAG','REFGRAD'});
    
    [vol, sens] = ft_prepare_vol_sens(vol, sens);
    mapping=[];
    for i=1:length(chanind)
        mapping(i)=find(strcmp(sens.label,Dnew.chanlabels{chanind(i)}));
    end
    
    if length(size(simsignal))==2
        tmp=zeros(length(chanind),Dnew.nsamples);
        for i=1:Ndip,
            gmn = ft_compute_leadfield(posdipmm(i,:)*1e-3, sens, vol,  'dipoleunit', 'nA*m','chanunit',sensorunits(sensorind));
            gain=gmn(mapping,:)*ordip(i,:)'*nAmdipmom(i);
            tmp(:,f1ind)=tmp(:,f1ind)+gain*simsignal(i,:);
        end; % for i
    else
        tmp=zeros(length(chanind),Dnew.nsamples,Dnew.ntrials);
        for i=1:Ndip,
            gmn = ft_compute_leadfield(posdipmm(i,:)*1e-3, sens, vol,  'dipoleunit', 'nA*m','chanunit',sensorunits(sensorind));
            gain=gmn(mapping,:)*ordip(i,:)'*nAmdipmom(i);
            for j=1:Dnew.ntrials
                tmp(:,f1ind,j)=tmp(:,f1ind,j)+gain*squeeze(simsignal(i,:,j));
            end
        end
    end
    
    
else %%% CURRENT DENSITY ON SURFACE SIMULATION
    disp('SIMULATING CURRENT DISTRIBUTIONS ON MESH');
    %% CREATE A NEW FORWARD model for e mesh
    fprintf('Computing Gain Matrix: ')
    spm_input('Creating gain matrix',1,'d');    % Shows gain matrix computation
    
    [L Dnew] = spm_eeg_lgainmat(Dnew);              % Gain matrix
    if isfield(Dnew.inv{val}.forward,'scale'),
        L=L./Dnew.inv{val}.forward.scale; %% account for rescaling of lead fields
    end;
    
    
    
    Nd    = size(L,2);                          % number of dipoles
    Nchans=size(L,1);
    
    fprintf(' - done\n')
    
    
    
    
    nativemesh=Dnew.inv{val}.forward.mesh;
    
    Qe=[];, %% SNR may be defined by sensor level data in which case we have to get data first then go back
    
    %[Qp,Qe,priors,priorfname] = spm_eeg_invert_EBconstruct_priors(Dnew,val,nativemesh,priors,Qe,L,'sim');
    base.FWHMmm=dipfwhm;
    base.nAm=nAmdipmom;
    
    
    [a1,b1,c1]=fileparts(Dnew.fname);
    
    priordir=[Dnew.path filesep 'simprior_' b1 ];
    mkdir(priordir);
    fprintf('Saving prior in directory %s\n',priordir);
    
    [Qp,Qe,priorfname]=spm_eeg_invert_setuppatches(meshsourceind,nativemesh,base,priordir,Qe,L);
    
    

    % Add waveform of all smoothed sources to their equivalent dipoles
    % QGs add up to 0.9854        
    
    X=zeros(size(full(Qp{1}.q)));
    if length(size(simsignal))==2
        fullsignal=zeros(Ndip,Dnew.nsamples); %% simulation padded with zeros
        fullsignal(1:Ndip,f1ind)=simsignal;
    
        tmp     = sparse(zeros(Nchans,Dnew.nsamples));                     % simulated data
        for j=1:Ndip
            Lq=L*Qp{j}.q; %% lead field * prior source distribution
            X=X+full(Qp{j}.q);
            for i=1:Dnew.nsamples,
                tmp(:,i) = tmp(:,i) + Lq*fullsignal(j,i); %% +sqrt(Qe)*randn(size(Qe,1),1);
            end;
        end;
    else
        fullsignal=zeros(Ndip,Dnew.nsamples,Dnew.ntrials); %% simulation padded with zeros
        fullsignal(1:Ndip,f1ind,:)=simsignal;
        tmp     = sparse(zeros(Nchans,Dnew.nsamples,Dnew.ntrials));                     % simulated data
        for j=1:Ndip
            Lq=L*Qp{j}.q; %% lead field * prior source distribution
            X=X+full(Qp{j}.q);
            for i=1:Dnew.nsamples,
                tmp(:,i,:) = tmp(:,i,:) + Lq*fullsignal(j,i,:); %% +sqrt(Qe)*randn(size(Qe,1),1);
            end;
        end;
    end
    tmp=tmp.*simscale; %% scale to sensor units
    
end; % if ori


allchanstd=std(full(tmp),[],2);
if length(size(simsignal))==3
    allchanstd=squeeze(mean(allchanstd,3));
end
meanrmssignal=mean(allchanstd);


if ~isempty(SNRdB),
    whitenoise = meanrmssignal.*(10^(-SNRdB/20));
    disp(sprintf('Setting white noise to give sensor level SNR of %dB',SNRdB));
end;

Qe=eye(Nchans).*(whitenoise^2); %% sensor level noise


YY=zeros(length(chanind),length(chanind));
n=0;
for i=1:Ntrials
    if any(i == trialind), %% only add signal to specific trials
        if length(size(tmp))==2
            Dnew(chanind,:,i) = full(tmp);
        else
            Dnew(chanind,:,i) = squeeze(tmp(:,:,i));
        end
    else
        Dnew(chanind,:,i)=zeros(size(tmp,1),size(tmp,2));
    end;
    Dnew(:,:,i)=Dnew(:,:,i)+randn(size(Dnew(:,:,i))).*whitenoise; %% add white noise in fT
    y=squeeze(Dnew(chanind,:,i));
    YY=YY+y*y';
    n=n+size(y,2); %% number of samples
end

%% Plot and save
[dum,tmpind]=sort(allchanstd);
dnewind=chanind(tmpind);

if isempty(ormni),
    YY=YY./n; %% NORMALIZE HERE

    F=[];
    UL=L;
    save(priorfname,'Qp','Qe','UL','F', spm_get_defaults('mat.format'));

    figure;
    ploton=1;
    [LQpL,Q,sumLQpL,QE,Csensor]=spm_eeg_assemble_priors(L,Qp,{Qe},ploton);


    figure;
    subplot(3,1,1);
    imagesc(YY);colorbar;
    title('Empirical data covariance per sample: YY');
    subplot(3,1,2);
    imagesc(Csensor);colorbar;
    title('Prior total sensor covariance');
    subplot(3,1,3);
    imagesc(YY-Csensor);colorbar;
    title('Difference');


    hold on
    mnivert=Dnew.inv{val}.mesh.tess_mni.vert;
    
    
    Nj      = size(mnivert,1);
    M       = X;
    G       = sqrt(sparse(1:Nj,1,abs(M),Nj,1)); %% ADDED IN ABS - NEED TO CHECK IN
    Fgraph  = spm_figure('GetWin','Graphics');
    j       = find(M);
    
    clf(Fgraph)
    figure(Fgraph)
    spm_mip(G(j),mnivert(j,:)',6);
    axis image
    title({sprintf('Generated source activity')});
    drawnow
end;
figure

if length(size(tmp))==2
    aux = tmp(tmpind(end),:);  
else
    aux = squeeze(mean(tmp(tmpind(end),:,:),3));
end 
subplot(2,1,1);
plot(Dnew.time,Dnew(dnewind(end),:,1),Dnew.time,aux,'r');
title('Measured activity over max sensor');
legend('Noisy','Noiseless');
ylabel(sensorunits{chanind(1)});
subplot(2,1,2);
if length(size(tmp))==2
    aux = tmp(tmpind(floor(length(tmpind)/2)),:);
else
    aux = squeeze(mean(tmp(tmpind(floor(length(tmpind)/2)),:,:),3));
end
plot(Dnew.time,Dnew(dnewind(floor(length(tmpind)/2)),:,1),Dnew.time,aux,'r');
title('Measured activity over median sensor');
legend('Noisy','Noiseless');
ylabel(sensorunits{chanind(1)});
xlabel('Time in sec');

Dnew.save;

fprintf('\n Finish\n')
