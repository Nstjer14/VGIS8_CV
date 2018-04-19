function P = urReadPosJ(sock)
% URREADPOSJ Reads the current joint values of UR robot.
%   P = URREADPOSJ(sock) uses an active socket connection to get the
%   joint values in [radians].
%
%   See also READPOSC, MOVEROBOT.


    if sock.BytesAvailable>0
        fscanf(sock,'%c',sock.BytesAvailable);
    end
    fprintf(sock,'(1)'); % task 1: Get joints
    while sock.BytesAvailable==0
    end
    rec = fscanf(sock,'%c',sock.BytesAvailable)
%     if ~strcmp(rec(1),'p') || ~strcmp(rec(end),']')
%         error('robotpose read error')
%     end
    
    rec(end) = ',';
    Curr_c = 1;
    for i = 1 : 6
        C = [];
        Done = 0;
        while(Done == 0)
            Curr_c = Curr_c + 1;
            if strcmp(rec(Curr_c) , ',')
                Done = 1;
            else
                C = [C,rec(Curr_c)]
            end
        end
        P(i) = str2double(C)  
    end
    
    for i = 1 : length(P)
        if isnan(P(i))
            error('robotpose read error (Nan)')
        end
    end
    
end
