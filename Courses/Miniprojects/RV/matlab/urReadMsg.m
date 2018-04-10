function msg = urReadMsg(sock)
% URREADMSG Reads message from a UR robot.
%   msg = URREADMSG(sock) is a low-level function which is intended to be
%   used through other functions like URMOVE.
%
%   See also URMOVE, URSETPARAM.

    msg = fscanf(sock,'%c',sock.BytesAvailable);
end