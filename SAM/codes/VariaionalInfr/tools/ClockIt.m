% function elapsedCpuTime = TimeIt(expression, n);
%
% Runs the matlab code in 'expression' (a string) for n iterations, and
% returns the total amount of CPU time used.  If n is omitted, it defaults to 1.
% 
% Example:
% x = rand(100, 100);
% ClockIt('NormalizeColumns(x)', 1000)
function elapsedCpuTime = TimeIt(expression, n);

if nargin < 2
  n = 1;
end

expression = [ sprintf('for asdfasdf = 1:%d ', n) expression '; end;' ]; 
t=cputime;
evalin('caller', expression);
elapsedCpuTime = cputime - t;
