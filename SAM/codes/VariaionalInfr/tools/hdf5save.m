% hdf5save.m
%
% hdf5save('filename','name in workspace','name in hdf5 file',...)
%
% Save the given variables in an hdf5 file. Syntax Similar to the native
% matlab save command. Can save strings, arrays, and structs. 
% 

% Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
% Copyright: Gael Varoquaux
% License: BSD-like

function hdf5save(filename,varargin)

% this is just a wrapper function that calls a recursive function that
% will save recursively structs.

nvars=size(varargin,2)/2;

if nvars~=floor(nvars) ;
    error('expecting a name for each variable') ;
end

%  Assign variables to save in workspace so as to be able to get them back
%  with recursive fonctions.
for i=[1:nvars]
    str=varargin{2*i};
    var=evalin('caller',str);
    assignin('base',str,var);
end

hdf5save_recursion(filename,'',1,varargin)

