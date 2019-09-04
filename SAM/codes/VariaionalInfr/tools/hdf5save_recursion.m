% hdf5save_recursion.m
%
% hdf5save('filename','prefix',new,'name in workspace','name in hdf5 file',...)

% Save the given variables in an hdf5 file in the groupe given by the
% prefix. If new is set to one creates a new hdf5 file. Syntax Similar to
% the native matlab save command. Can save strings, arrays, and structs.
% If it find a struct it calls its recusively to save the struct, passing
% the struct name as the prefix.

% Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
% Copyright: Gael Varoquaux
% License: BSD-like

function hdf5save_recursion(filename,prefix,new,varcell)

if ~strcmp(class(filename),'char') ;
    error('first argument should be a string giving the path to the hdf5 file to save') ;
end

if ~strcmp(class(prefix),'char') ;
    error('second argument should be a string giving the name of the groupe to save the data in') ;
end

if new~=1 && new~=0 ;
    error('third argument should be 0 or 1: 0 to append data, and 1 to create a new file')
end

nvars=size(varcell,2)/2;

if nvars~=floor(nvars) ;
    error('expecting a name for each variable') ;
end

for i=[1:nvars]
    str=varcell{2*i};
    var=evalin('base',str);
    name=varcell{2*i-1};
    type=class(var);
    location=strcat('/',prefix,name);
    %disp(['variable name in workspace : ',str]);
    %disp(['variable name to put : ',name]);
    switch type
    case 'struct'
	names=fieldnames(var);
	for j=1:size(names,1) ;
	    if (j~=1 || i~=1)
		new=0 ;
	    end
	    varname=strcat(name,'.',names{j});
	    hdf5save_recursion(filename,strcat(name,'/'),new,{names{j},varname});
	end
    otherwise
	location=strcat('/',prefix,name);
	if new==1 && i==1 ;
	    hdf5write(filename,location,var);
	else
	    %disp(['location : ',location]);
	    hdf5write(filename,location,var,'WriteMode', 'append');
	end
    end
end

