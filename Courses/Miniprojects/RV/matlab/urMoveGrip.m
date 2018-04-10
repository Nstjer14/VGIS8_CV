function urMoveGrip(sock, closeGrip)
% URMOVEGRIP Opens/closes gripper based on boolean value of closeGrip.
%   URMOVEGRIP(sock, closeGrip) works for OnRobot RG2 gripper.
%
%   Note ... This function is not updated!!!
    
   if closeGrip
       fprintf(sock,'(6)')
   else
       fprintf(sock,'(5)')
   end

end

