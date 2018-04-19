function P = urReadPosC(sock)
% URREADPOSC Reads the current pose of a UR robot.
%   P = URREADPOSC(sock) uses an active socket connection to get the
%   pose and return it in the form [tx, ty, tx, r1, r2, r3]. The
%   translations are in [mm] and the axis-angle rotations are in [radians].
%
%   See also READPOSJ, MOVEROBOT.

    if sock.BytesAvailable>0
        fscanf(sock,'%c',sock.BytesAvailable);
    end
    fprintf(sock,'(2)'); % task 2: Get pose
    while sock.BytesAvailable==0
    end
    rec = fscanf(sock,'%c',sock.BytesAvailable);
    if ~strcmp(rec(1),'p') || ~strcmp(rec(end),']')
        error('robotpose read error')
    end
    
    rec(end) = ',';
    Curr_c = 2;
    for i = 1 : 6
        C = [];
        Done = 0;
        while(Done == 0)
            Curr_c = Curr_c + 1;
            if strcmp(rec(Curr_c) , ',')
                Done = 1;
            else
                C = [C,rec(Curr_c)];
            end
        end
        P(i) = str2double(C);   
    end
    
    for i = 1 : length(P)
        if isnan(P(i))
            error('robotpose read error (Nan)')
        end
    end
    P(1:3) = P(1:3)*1000; % converting to mm
end
