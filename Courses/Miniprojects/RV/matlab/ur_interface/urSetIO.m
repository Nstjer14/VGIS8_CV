function urSetIO(sock, ioNum, ioState)
% URSETIO 
%   URSETIO(sock, moveType) uses an active socket connection.
%
%   'ioState' is the IO-value (0 or 1).
%   'ioNum' is thenumber of the digital_output.
%
%   See also URCHANGEGRIP, URCHANGEVEL, URREADPOSE, URMOVE.

if nargin <= 2
    error('error; not enough input arguments')
elseif nargin >= 4
    error('error; too many input arguments')
end
urSetParam(sock, 15, [ioState ioNum]); %task=15: change state of an IO port
