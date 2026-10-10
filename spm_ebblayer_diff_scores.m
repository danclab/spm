function [scores, relative_separation, rejected] = spm_ebblayer_diff_scores(layer_leads, InvCov, layer_pairs, tiny, min_relative_norm)
%SPM_EBBLAYER_DIFF_SCORES Numerically stable DIFF evidence for laminar EBB.
%
% [SCORES, RELATIVE_SEPARATION, REJECTED] = ...
%       spm_ebblayer_diff_scores(LEADS, INVCOV, PAIRS, TINY, MIN_REL)
%
% LEADS has spatial-modes x layers dimensions; PAIRS is a nPairs x 2 matrix
% of MATLAB one-based layer indices.  Each difference is constructed in
% sensor space BEFORE taking its quadratic forms:
%
%   delta = LEADS(:,a) - LEADS(:,b)
%   score = (delta'*delta) / (4*delta'*INVCOV*delta).
%
% The source difference is discarded if either quadratic form is too small
% or nonfinite, or if ||delta||/max(||a||,||b||) < MIN_REL.  The latter is
% a numerical guard, not a threshold on physical laminar identifiability.
% A zero score is ineligible for subsequent top-K selection.
%
% This helper is called once per cortical column, not once per pair.
%
% See also SPM_EEG_INVERT_CLASSIC.

if nargin ~= 5
    error('spm_ebblayer_diff_scores requires five input arguments.');
end

[nmodes, nlayers] = size(layer_leads);
if ~isnumeric(layer_leads) || ~isreal(layer_leads) || ...
        ~isnumeric(InvCov) || ~isequal(size(InvCov), [nmodes nmodes]) || ...
        ~isreal(InvCov) || ...
        ~isnumeric(layer_pairs) || size(layer_pairs,2) ~= 2 || ...
        any(layer_pairs(:) < 1) || any(layer_pairs(:) > nlayers) || ...
        any(layer_pairs(:) ~= round(layer_pairs(:)))
    error('Incompatible lead fields, covariance, or layer-pair indices.');
end
if ~isscalar(tiny) || ~isfinite(tiny) || tiny < 0 || ...
        ~isscalar(min_relative_norm) || ~isfinite(min_relative_norm) || ...
        min_relative_norm < 0
    error('Tolerances must be finite nonnegative scalars.');
end

npairs = size(layer_pairs,1);
a = layer_pairs(:,1);
b = layer_pairs(:,2);

% Form differences in sensor space first.  Gram-matrix differences such as
% (a'*a + b'*b - 2*a'*b) lose all significant digits for a ~= b when
% the relative difference is close to sqrt(eps).
delta = layer_leads(:,a) - layer_leads(:,b);
den = sum(delta .* delta, 1);
num = sum(delta .* (InvCov * delta), 1);

% Dimensionless separation, evaluated without subtracting Gram products.
% Use hypot-style norms to avoid accidentally dividing by a zero column.
na = sqrt(sum(layer_leads(:,a).^2, 1));
nb = sqrt(sum(layer_leads(:,b).^2, 1));
ref = max(na, nb);
relative_separation = zeros(1,npairs);
valid_ref = isfinite(ref) & ref > 0;
relative_separation(valid_ref) = sqrt(den(valid_ref)) ./ ref(valid_ref);

eligible = valid_ref & isfinite(den) & isfinite(num) & ...
    isfinite(relative_separation) & den > tiny & num > tiny & ...
    relative_separation >= min_relative_norm;

scores = zeros(1,npairs);
scores(eligible) = den(eligible) ./ (4 * num(eligible));
scores(~isfinite(scores) | scores <= 0) = 0;

% Count only pairs rejected by the relative-separation criterion, where
% both underlying quadratic forms would otherwise have been valid.
base_valid = valid_ref & isfinite(den) & isfinite(num) & ...
    den > tiny & num > tiny;
rejected = base_valid & ~eligible;
end
