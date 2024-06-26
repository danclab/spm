function bf = tbx_cfg_bf
% Configuration file for toolbox 'Beamforming'
%__________________________________________________________________________

% Vladimir Litvak
% Copyright (C) 2012-2023 Wellcome Centre for Human Neuroimaging


tbxdir = fileparts(mfilename('fullpath'));

if ~isdeployed, addpath(tbxdir); end

components = {
    'bf_group';
    'bf_data';
    'bf_copy';
    'bf_sources'
    'bf_features'
    'bf_inverse'
    'bf_output'
    'bf_write'
    'bf_view'
    };

bf = cfg_choice;
bf.tag = 'beamforming';
bf.name = 'DAiSS (beamforming)';
bf.help = {'Data analysis in source space toolbox'};

for i = 1:numel(components)
  bf.values{i} = feval(components{i});
end

% Generate the menu function automatically in case of a different directory
% name (might fail if there is no write permission)
[tbx_path, tbx_name] = fileparts(tbxdir);
if ~isdeployed && ~isequal(tbx_name, 'beamforming')
    if ~exist(fullfile(tbxdir, ['spm_' tbx_name '.m']), 'file')
        try
            fid = fopen(fullfile(tbxdir, ['spm_' tbx_name '.m']), 'wt');
            fprintf(fid, 'function spm_%s\n\nspm_beamforming', tbx_name);
            fclose(fid);
        end
    end
end
