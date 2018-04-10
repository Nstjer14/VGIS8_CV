function urSetParam(sock, task, param_vec)
% URSETPARAM 
%   pose_new = URSETPARAM(sock, task, param_vec) uses an active socket
%   connection to set a parameter for a specific task.
%   'task' is a one-digit integer
%   'param_vec' is a vector of integers with max length 6.
%
%   URSETPARAM is a low-level function which is intended to be used through
%   other functions like URCHANGEMOVE.
%
%   See also URMOVE, URCHANGEMOVE, URCHANGEVEL, URCHANGEGRIP, URTOGGLEGRIP.

success = '0';

if nargin < 3
    error('error; not enough input arguments');
end

% Create data structure:
data = zeros(1,6);
for i = 1:length(param_vec)
    data(i) = param_vec(i);
end
param_str = ['(',...
    num2str(data(1)),',',...
    num2str(data(2)),',',...
    num2str(data(3)),',',...
    num2str(data(4)),',',...
    num2str(data(5)),',',...
    num2str(data(6)),...
    ')'];

while strcmp(success,'0')
    task_str = ['(' num2str(task) ')'];
    fprintf(sock, task_str);
    pause(0.01);% Tune this to meet your system
    fprintf(sock, param_str);
    while sock.BytesAvailable==0
        %t.BytesAvailable
    end
    
    success = urReadMsg(sock);
end
if ~strcmp(success,'1')
    error('error communicating with robot')
end
%pause(0.5)
% pose_new = urReadPose(sock);


