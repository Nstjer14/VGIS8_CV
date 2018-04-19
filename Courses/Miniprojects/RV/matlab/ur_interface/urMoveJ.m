function urMoveJ(sock, joints)
% URMOVEJ Moves a UR robot to a new joint configuration.
%   URMOVEJ(sock, pose_goal, orientation) uses an active
%   socket connection to move the robot to a new pose. 
%
%   'joints' specify the new joint values in [radians].
%
%   See also URMOVEL, URMOVEP, URREADPOSE, URMOVETRANS.

if nargin ~= 2
    error('error; wrong number of input arguments')
end

urSetParam(sock, 11, joints); % task 11: MoveJ

success = '0';
while strcmp(success,'0')
    while sock.BytesAvailable==0
        %t.BytesAvailable
    end
    success  = urReadMsg(sock);
end
