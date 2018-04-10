function urChangeGrip(sock, gripVar)
% URCHANGEGRIP Controls the OnRobot RG2 gripper
%   pose_new = URCHANGEGRIP(sock,gripVar) uses an active socket connection.
%   gripVar is a 4-vector.
%
%   See also URCHANGEMOVE, URREADPOSE, URMOVE.

if nargin == 1
    error('error; not enough input arguments')
elseif nargin == 3
    error('error; too many input arguments')
end
urSetParam(sock, 18, gripVar); % task 18: setRG2