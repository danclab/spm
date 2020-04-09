function cfg = tbx_cfg_mb

if ~isdeployed, addpath(fullfile(spm('dir'),'toolbox','mb')); end

images        = cfg_files;
images.tag    = 'images';
images.name   = 'Scans';
images.filter = 'nifti';
images.num    = [1 Inf];
images.help   = {['Select one NIfTI format scan for each subject. Subjects must be in the same order if there are multiple channels. '...
                  'Image dimensions can differ over subjects, but (if there are multiple channels) the scans of each '...
                  'subject must all have the same dimensions and orientations.'],''};

cm_map        = cfg_entry;
cm_map.tag    = 'cm_map';
cm_map.name   = 'Row';
cm_map.strtype = 'n';
cm_map.num    = [1 Inf];
cm_map.help   = {'For this value in the label map, specify which tissue classes it can correspond to.',''};

cm        = cfg_repeat;
cm.tag    = 'cm';
cm.name   = 'Confusion matrix';
cm.values = {cm_map};
%cm.val   = {};
cm.help   = {'Specify rows of a confusion matrix, where each row corresponds to label values of 1, 2, ..., etc in the label maps.',''};


inu_reg      = cfg_entry;
inu_reg.tag  = 'inu_reg';
inu_reg.name = 'Regularisation';
inu_reg.strtype = 'e';
inu_reg.num  = [1 1];
inu_reg.val  = {1e5};
inu_reg.help = {['Specify the bending energy penalty on the estimated intensity nonuniformity (INU) '...
                'fields (bias fields). Larger values give smoother INU fields.'],''};

inu_co        = cfg_menu;
inu_co.tag    = 'inu_co';
inu_co.name   = 'Cut off';
inu_co.labels = {' 20 mm INU', ' 40 mm INU', ' 60 mm INU', ' 80 mm INU', '100 mm INU', 'Rescale only', 'No correction'};
inu_co.values = {20, 40, 60, 80, 100, Inf, NaN};
inu_co.val    = {60};
inu_co.help   = {['Specify the cutoff (mm) of the intensity nonuniformity (INU) correction (bias correction). '...
                 'Larger values use fewer parameters to encode the INU field. '...
                 'Note that a global intensity rescaling correction, without INU correction, can also be specified. '...
                 'For quantitative images, it may be better not to use any correction.'],''};

inu           = cfg_branch;
inu.tag       = 'inu';
inu.name      = 'Intensity nonuniformity';
inu.val       = {inu_reg,inu_co};
inu.help      = {'Intensity nonuniformity (INU) settings.',''};


label_files        = cfg_files;
label_files.tag    = 'images';
label_files.name   = 'Label maps';
label_files.filter = 'nifti';
label_files.num    = [1 Inf];
label_files.help   = {['Label maps are NIfTI images containing integer values, which must have the same '...
                       'dimensions and orientations as the scans of the corresponding subjects. '...
                       'Voxels of each value in the label map may be included in one or more tissue classes. '...
                       'For example, a label map showing the location of brain will include voxels that can '...
                       'be in grey or white matter classes.'],''};

label_pr        = cfg_const;
label_pr.tag    = 'w';
label_pr.name   = 'Confidence';
label_pr.val    = {0.99};
label_pr.hidden = true;
label_pr.help   = {'Degree of confidence in the labels.',''};

labels         = cfg_branch;
labels.tag     = 'true';
labels.name    = 'Yes';
labels.val     = {label_files,cm,label_pr};
labels.help    = {'Subjects have corresponding label maps to guide the segmentation.',''};

no_labels      = cfg_const;
no_labels.tag  = 'false';
no_labels.name = 'No';
no_labels.val  = {[]};
no_labels.help = {'Subjects do not have corresponding label maps.',''};

has_labels        = cfg_choice;
has_labels.tag    = 'labels';
has_labels.name   = 'Labels?';
has_labels.values = {labels,no_labels};
has_labels.val    = {no_labels};
has_labels.help   = {'Specify whether there are pre-defined label maps for the subjects.',''};

modality      = cfg_menu;
modality.tag  = 'modality';
modality.name = 'Modality';
modality.labels = {'MRI','CT'};
modality.values = {1,2};
modality.val    = {1};
modality.help   = {'Specify the modality of the scans in this channel.',''};

chan          = cfg_branch;
chan.tag      = 'chan';
chan.name     = 'Channel';
chan.val      = {images, inu, modality};
chan.help     = {['There may be multiple scans of different modalities for each subject. '...
                  'These would be entered into different channels. Note that all scans '...
                  'within a channel should be the same modality.'],''};

chans         = cfg_repeat;
chans.tag     = 'chans';
chans.name    = 'Channels';
chans.values  = {chan};
chans.num     = [1 Inf];
chans.val     = {chan};
chans.help    = {['Multiple image channels may be specified. For example, two channels may be used to contain '...
                  'the T2-weighted and PD-weighted scans of the subjects.'],''};

pr_dat         = cfg_files;
pr_dat.tag     = 'file';
pr_dat.name    = 'Definition';
pr_dat.filter  = 'mat';
pr_dat.ufilter = '^prior.*\.mat';
pr_dat.dir     = datadir;
pr_dat.num     = [0 1];
pr_dat.val     = {{}};
pr_dat.help    = {['Knowledge of Gaussian-Wishart priors for the intensity distributions of each cluster '...
                   'can help to inform the segmentation. When available, this information is specified in MATLAB prior*.m files. '...
                   'These files currently need to be hand-crafted. Unless you understand what you are doing, '...
                   'it is advised that you do not specify and intensity prior definition.'],''};

pr_upd         = cfg_menu;
pr_upd.tag     = 'update';
pr_upd.name    = 'Optimise';
pr_upd.labels  = {'Yes','No'};
pr_upd.values  = {true, false};
pr_upd.val     = {true};
pr_upd.help    = {['Specify whether the Gaussian-Wishart priors be updated at each iteration. '...
                   'Enabling this can slow down convergence if there are small numbers of subjects. '...
                   'If only one subject is to be modelled (using a pre-computed template), then '...
                   'definitely turn off this option.'],''};

pr       = cfg_branch;
pr.tag   = 'pr';
pr.name  = 'Intensity prior';
pr.val   = {pr_dat,pr_upd};
pr.help  = {['Intensity distributions of each tissue class are modelled by a Gaussian distribution. '...
             'Prior knowledge about these distributions can make the model fitting more robust.'],''};

pop       = cfg_branch;
pop.tag   = 'gmm';
pop.name  = 'Pop. of scans';
pop.val   = {chans, has_labels, pr,...
             const('tol_gmm', 5e-5), const('nit_gmm_miss',32), const('nit_gmm',16), const('nit_appear', 4)};
pop.check = @check_pop;
%pop.val  = {chans};
pop.help  = {'Information about a population of subjects that all have the same set of scans.',''};

pops            = cfg_repeat;
pops.tag        = 'pops';
pops.name       = 'Populations';
pops.values     = {pop};
pops.num        = [0 Inf];
pops.val        = {};
pops.help       = {['Multiple populations of subjects may be combined. For example, there may be '...
                    'T1-weighted scans and manually defined labels for one population, whereas '...
                    'another population may have T2-weighted and PD-weighted scans without labels. '...
                    'Yet another population might have CT scans. All subject''s data would be '...
                    'subdivided into the same tissue classes, although the intensity distributions '...
                    'of these tissues is likely to differ accross populations.'],''};

seg          = cfg_files;
seg.tag      = 'cat';
seg.name     = 'Class';
seg.filter   = 'nifti';
seg.ufilter  = '.*c[0-9].*';
seg.num      = [0 Inf];
seg.val      = {{}};
seg.help     = {'Tissue class images of the same class and multiple subjects produced by some previously run segmentation method.',''};

segs         = cfg_repeat;
segs.tag     = 'images';
segs.name    = 'Classes';
segs.values  = {seg};
segs.val     = {seg};
segs.help    = {['Images might have been segmented previously into a number of tissue classes.'...
                 'This framework allows such pre-segmented images to be included in the model fitting, '...
                 'in a similar way to the old Dartel toolbox for SPM.'],''};

%is_imp        = cfg_menu;
%is_imp.tag    = 'mat0';
%is_imp.name   = 'Imported?';
%is_imp.labels = {'no','yes'};
%is_imp.values = {false,true};
%is_imp.val    = {false};
%is_imp.help   = {['This option is included for backward compatibility with the old Dartel and Shoot toolboxes. '...
%                  'Because rigid-body alignment is included within the model, images do not actually need to be in '...
%                  'rigid alignment beforehand.'],''};

spop          = cfg_branch;
spop.tag      = 'cat';
spop.name     = 'Tissue class maps';
spop.val      = {images};
spop.check    = @check_segs;
spop.help     = {'Specify the data to be included within the model.',''};


mu_exist        = cfg_files;
mu_exist.tag    = 'exist';
mu_exist.name   = 'Existing template';
mu_exist.filter = 'nifti';
mu_exist.dir    = datadir;
mu_exist.num    = [1 1];
mu_exist.help   = {['The model can be fit using a previously computed template, which is not updated. '...
                    'Note that the template contains K-1 volumes within it, and that K should be compatible '...
                    'with various aspects of the data to which the model is fit. The template does not '...
                    'actually encode the tissue probabilities, but rather these probabilities can be generated '...
                    'from the template using a Softmax function/* (${\bf p} = \frac{\exp {\bf p}}{1 + \sum_k \exp p_k} )*/.'],''};


nclass          = cfg_entry;
nclass.tag      = 'K';
nclass.name     = 'Number of classes';
nclass.strtype  = 'e';
nclass.val      = {9};
nclass.num      = [1 1];
nclass.help     = {['Specify K, the number of tissue classes encoded by the template. '...
                    'This value is ignored if it is incompatible with the specified data.'],''};

vox          = cfg_entry;
vox.tag      = 'vx';
vox.name     = 'Voxel size';
vox.strtype  = 'e';
vox.val      = {1};
vox.num      = [1 1];
vox.help     = {'Specify the voxel size of the template (mm). ',''};


mu_sett        = cfg_const;
mu_sett.tag    = 'mu_settings';
mu_sett.name   = 'Mu settings';
mu_sett.val    = {[1.0000e-05 0.5000 0]};
mu_sett.hidden = true;


mu_create       = cfg_branch;
mu_create.tag   = 'create';
mu_create.name  = 'Create template';
mu_create.val   = {nclass, vox, mu_sett};
mu_create.help  = {['A tissue probability template will be constructed from all the aligned images. '...
                    'The algorithm alternates between re-computing the template and re-aligning all the '...
                    'images with this template.']};

mu_prov         = cfg_choice;
mu_prov.tag     = 'mu';
mu_prov.name    = 'Template';
mu_prov.values  = {mu_create, mu_exist};
mu_prov.val     = {mu_create};
mu_prov.help    = {'The model can be run using a pre-computed template, or it can implicitly compute its own template.',''};

aff             = cfg_menu;
aff.tag         = 'aff';
aff.name        = 'Affine';
aff.labels      = {'None', 'Translations', 'Rigid'};
aff.values      = {'', 'T(3)', 'SE(3)'};
aff.val         = {'SE(3)'};
aff.help        = {'Type of affine transform to use in the model.',''};

dff             = cfg_entry;
dff.tag         = 'v_settings';
dff.name        = 'Shape regularisation';
dff.strtype     = 'e';
dff.num         = [1 5];
dff.val         = {[0.0001 0 0.5 0.125 0.5]};
dff.help        = {'Regularisation settings for the diffeomorphic registration. The defaults work reasonably well.',''};

odir            = cfg_files;
odir.tag        = 'odir';
odir.name       = 'Output directory';
odir.filter     = 'dir';
odir.num        = [1 1];
odir.val        = {{'.'}};
odir.help       = {'All output is written to the specified directory. The current working directory is used by default.',''};

onam            = cfg_entry;
onam.tag        = 'onam';
onam.name       = 'Output name';
onam.strtype    = 's';
onam.val        = {'mb'};
onam.help       = {'A key string may be included within all the output files.',''};

mb             = cfg_exbranch;
mb.tag         = 'run';
mb.name        = 'Fit Multi-Brain model';
mb.val         = {mu_prov, aff, dff, onam, odir, segs, pops, ...
                   const('accel',0.8), const('min_dim', 10), const('tol',2e-4), const('sampdens',2),const('save',true),const('nworker',0)};
mb.prog        = @run_mb;
mb.help        = {['This framework attempts to unify ``unified segmentation'''' with ``shoot'''', '...
                    'as well as a great deal of other functionality. '...
                    'It has the general aim of integrating a number of disparate image analysis '...
                    'components within a single unified generative modelling framework. '...
                    'The objective is to achieve diffeomorphic alignment of a wide variaty of medical '...
                    'image modalities into a common anatomical space. This involves the ability to construct '...
                    'a ``tissue probability template'''' from a population of scans through group-wise '...
                    'alignment/* \cite{john_averageshape,blaiotta2018generative}*/, which incorporates '...
                    'both rigid and diffeomorphic registration/* \cite{ashburner2013symmetric}*/. Diffeomorphic '...
                    'deformations are computed within a geodesic shooting framework/* \cite{ashburner2011diffeomorphic}*/, '...
                    'which is optimised with a Gauss-Newton strategy that uses a multi-grid approach to '...
                    'solve the system of linear equations/* \cite{ashburner07}*/. Variability among image '...
                    'contrasts is modelled using a much more sophisticated version of the Gaussian mixture '...
                    'model with an intensity nonuniformity (INU) correction framework originally proposed by Ashburner & Friston/* \cite{ashburner05}*/, '...
                    'and which has been extended to account for known variability of the intensity '...
                    'distributions of different tissues/* \cite{blaiotta2016variational,blaiotta2018generative,brudfors2019empirical}*/. '...
                    'This model has been shown to provide a good model of the intensity distributions of '...
                    'different imaging modalities/* \cite{brudfors2019empirical}*/. Time permitting, additional '...
                    'registration accuracy through the use of shape variability priors/* \cite{balbastre2018diffeomorphic}*/ '...
                    'will also be incorporated.'],...
                    'This work was funded by the EU Human Brain Project’s Grant Agreement No 785907 (SGA2).',...
                    ''};




res_file        = cfg_files;
res_file.tag    = 'result';
res_file.name   = 'MB results file';
res_file.filter = '^mb_fit.*';
res_file.num    = [1 1];
res_file.help   = {'Specify the results file from the groupwise alignment.',''};

ix        = cfg_entry;
ix.tag    = 'ix';
ix.name   = 'Indices';
ix.strtype = 'n';
ix.num    = [1 Inf];
ix.help   = {['Specify indices. For example, if the original model '...
              'had K=9 and you wish to combine the final three classes, '...
              'then enter 1 2 3 4 5 6 7 7 7. Note that K refers to the '...
              'total number of tissue maps -- including the implicit '...
              'background.'],''};

onam.val = {'merged'};

mrg      = cfg_exbranch;
mrg.tag  = 'merge';
mrg.name = 'Merge tissues';
mrg.val  = {res_file, ix, onam,odir};
mrg.prog = @spm_mb_merge;
mrg.help = {['Merge tissues together and extract intensity priors '...
             'for later use.'],''};

cfg        = cfg_repeat;
cfg.tag    = 'mb';
cfg.name   = 'Multi-Brain toolbox';
cfg.values = {mb,mrg};
%cfg.val   = {};
cfg.help   = {'Welcome to the Multi-Brain toolbox.',''};


function cfg = const(tag,val)
cfg        = cfg_const;
cfg.tag    = tag;
cfg.val    = {val};
cfg.hidden = true;


function out   = run_mb(cfg)
out = 'config.mat';
save(out,'cfg');
[dat,sett] = spm_mb_init(cfg);
save(out,'cfg','dat','sett');
[dat,sett,mu] = spm_mb_fit(dat,sett);
%save(out,'cfg','dat','sett');
dat = spm_mb_io('SavePsi',dat,sett);
save(fullfile(sett.odir,['mb_fit_' sett.onam '.mat']),'sett','dat');


function dr = datadir
fullname  = mfilename('fullpath');
pth       = fileparts(fullname);
dr        = fullfile(pth,'data');


function str = check_pop(cfg)
str = '';
N = -1;
for c=1:numel(cfg.chan)
    Nc = numel(cfg.chan(c).images);
    if N<0
        N = Nc;
    else
        if N~=Nc
            str = 'Incompatible numbers of scans over channels.';
        end
    end
end
if isfield(cfg.labels,'true')
    Nc = numel(cfg.chan(c).images);
    if N~=Nc
        str = 'Incompatible numbers of label images.';
    end
end

function str = check_segs(cfg)
str    = '';
N      = -1;
images = cfg.images;
for c=1:numel(images)
    Nc = numel(images{c});
    if N<0
        N = Nc;
    else
        if N~=Nc
            str = 'Incompatible numbers of categorical images.';
        end
    end
end
