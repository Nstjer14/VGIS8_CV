function urMoveTrans(sock, is_absolute, trans)
% URMOVETRANS Moves a UR robot to a relative or absolute position.
%   P_new = URMOVETRANS(sock, is_absolute, trans) uses an active socket
%   connection to move the robot to a new pose. trans is a 3-vector
%   translation in [mm], and is_absolute controls if the translation is
%   absolute or relative to the current translation.
%
%   See also URMOVEROT, URMOVE.

    Robot_Pose = urReadPose(sock);
    Translation = Robot_Pose(1:3); % in mm
    Orientation = Robot_Pose(4:6);

    if is_absolute
        Translation = trans;
    else
        Translation = Translation + trans;
    end
    urMoveL(sock,Translation,Orientation);

end

