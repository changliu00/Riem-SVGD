% function result = IsOctave();
% Determines (hackily) whether this interpreter is octave.
function result = IsOctave();
result = ~isempty(which('octave_config_info'));