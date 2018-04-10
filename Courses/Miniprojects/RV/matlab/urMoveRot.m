function urMoveRot(sock, is_absolute, axis, angle_deg)
% URMOVEROT Moves a UR robot to a relative or absolute position.
%   P_new = URMOVEROT(sock, is_absolute, trans) uses an active socket
%   connection to move the robot to a new pose.
%
%   'axis' is a 3-vector rotation vector,
%   'angle_deg' is the angle to rotate around this vector, and
%   'is_absolute' controls if the rotation is absolute or relative to
%   the current oriantation.
%
%   See also URMOVETRANS, URMOVE.
    
    rad = angle_deg * pi/180;
    Robot_Pose = urReadPose(sock);
    Translation = Robot_Pose(1:3); % in mm
    Orientation = Robot_Pose(4:6);
    
    if is_absolute
        Goal_ori = axis*rad;
    else
        orientation_mat = vrrotvec2mat([Orientation,norm(Orientation)]);
        Rot_mag = rad;
        Rot_z = vrrotvec2mat([axis Rot_mag]);
        Goal_orient = orientation_mat *Rot_z;
    
        Goal_v = vrrotmat2vec(Goal_orient(1:3,1:3));
        Goal_ori = Goal_v(4)*Goal_v(1:3);
    end
    
    urMoveL(sock, Translation, Goal_ori);
end

