function urMoveL(sock, pose_goal, orientation)
% URMOVEL Moves a UR robot linearly to a new pose.
%   URMOVEL(sock, pose_goal, orientation) uses an active
%   socket connection to move the robot to a new pose. pose_goal is a
%   3-vector translation in [mm], and orientation is a 3-vector axis-angle
%   orientation in [radians].
%
%   URMOVEL(sock, pose_goal) combines translation and
%   orientation into a 6-vector [tx, ty, tx, r1, r2, r3].
%
%   See also URMOVEP, URMOVEJ, URREADPOSE, URMOVETRANS.

if nargin == 1
    error('error; not enough input arguments')
elseif nargin == 2
    P = pose_goal;
elseif nargin == 3
    P = [pose_goal,orientation];
end
P(1:3) = P(1:3) * 0.001; % Converting to meter
urSetParam(sock, 13, P); % task 13: MoveJ

success = '0';
while strcmp(success,'0')
    while sock.BytesAvailable==0
        %t.BytesAvailable
    end
    success  = urReadMsg(sock);
end


