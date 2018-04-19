function urChangeVel(sock, speedVec)
% URCHANGEVEL
%   pose_new = URCHANGEVEL(sock,speedVec) uses an active socket connection.
%   speedVec is a 2-vector which sets [vel, acc] on the UR robot.
%   The units are vel[m/s] and acc[m/s^2] for Cartesian movements and
%   vel[rad/s] and acc[rad/s^2] for Joint movements.
%
%   Default values are:
%   vel = 0.2rad/s    or  vel = 0.2m/s
%   acc = 0.2rad/s^2  or  acc = 0.2m/s^2
%
%   See also URCHANGEMOVE, URMOVE.

if nargin == 1
    error('error; not enough input arguments')
elseif nargin == 2
    S = speedVec;
elseif nargin == 3
    error('error; too many input arguments')
end

urSetParam(sock, 14, speedVec); % task 14: Change speed
