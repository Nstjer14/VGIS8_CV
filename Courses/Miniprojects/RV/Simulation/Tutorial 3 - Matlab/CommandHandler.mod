MODULE CommandHandler

PROC COMH_CommandExecutor(string command, string data)
    
    VAR jointtarget jointValues;
    VAR pos varPos;
    VAR orient varOrient;
    VAR speeddata varSpeed;
    VAR num objectedGrasped;
    
    IF command = "GETPOS" THEN
        
        FH_GetPosition varPos, varOrient;
        CONH_SendData "["+ValToStr(varPos.x)+","+ValToStr(varPos.y)+","+ValToStr(varPos.z)+","+ValToStr(varOrient.q1)+","+ValToStr(varOrient.q2)+","+ValToStr(varOrient.q3)+","+ValToStr(varOrient.q4)+"]";       
        RETURN;   
        
    ELSEIF command = "GETJOINTVALUES" THEN
    
        FH_GetJointValues jointValues;
        CONH_SendData ValToStr(jointValues.robax);
        RETURN;
        
    ENDIF
    
    CONH_SendData command+";started";
    
    IF command = "GRIPPERON" THEN
        
        FH_GripperActivation 1,objectedGrasped;
        CONH_SendData command+";ended;"+ValToStr(objectedGrasped);
        RETURN;
        
    ELSEIF command = "GRIPPEROFF" THEN
        
        FH_GripperActivation 0,objectedGrasped;
        
    ELSEIF command = "MOVEJOINTS" THEN
        
        COMH_JointDataEnterpretor data, jointValues, varSpeed;
        FH_MoveJoints jointValues, varSpeed;
    
    ELSEIF command = "MOVEPTP" THEN
        
        COMH_MovementDataEnterpretor data, varPos, varOrient, varSpeed;
        FH_MovePTP varPos, varOrient, varSpeed;
        
    ELSEIF command = "MOVEL" THEN
        
        COMH_MovementDataEnterpretor data, varPos, varOrient, varSpeed;
        FH_MoveLinear varPos, varOrient, varSpeed;
                
    ELSEIF command = "MOVEPTPR" THEN
        
        COMH_MovementDataEnterpretor data, varPos, varOrient, varSpeed;
        FH_MoveRelativePTP varPos, varOrient, varSpeed;
    
    ELSEIF command = "MOVELR" THEN
        
        COMH_MovementDataEnterpretor data, varPos, varOrient, varSpeed;
        FH_MoveRelativeLinear varPos, varOrient, varSpeed;
        
    ELSEIF command = "TAKEPICTURE" THEN
        
        FH_TakePicture;
        
    ENDIF
    
    CONH_SendData command+";ended";
        
ENDPROC
    
PROC COMH_CommandListener()
    
    VAR string recieve_string; 
    VAR string command;
    VAR string data;
        
    CONH_RecieveData recieve_string;
    COMH_SplitString recieve_string, command, data;
    COMH_CommandExecutor command, data;
    
ENDPROC

PROC COMH_JointDataEnterpretor(string data, INOUT jointtarget jointValues, INOUT speeddata moveSpeed)
    
    VAR bool dummy;
    VAR string jointData;
    VAR string speed;
    
    COMH_SplitString data, jointData, speed;
    
    dummy := StrToVal(jointData,jointValues.robax);
    COMH_SpeedDataEnterpretor speed, moveSpeed;
    
ENDPROC

PROC COMH_MovementDataEnterpretor(string data, INOUT pos movePos, INOUT orient moveOrient, INOUT speeddata moveSpeed)
    
    VAR string tempString;
    VAR string posdata;
    VAR string orientdata;
    VAR string speed;
    VAR bool dummy;
    
    COMH_SplitString data, posdata, tempString;
    COMH_SplitString tempString, orientdata, speed;
    
    dummy := StrToVal(posdata, movePos);
    dummy := StrToVal(orientdata, moveOrient);
    COMH_SpeedDataEnterpretor speed, moveSpeed;
    
ENDPROC

PROC COMH_SpeedDataEnterpretor(string speed, INOUT speeddata moveSpeed)

    IF speed = "v5" THEN
        moveSpeed := v5;
    ELSEIF speed = "v10" THEN
        moveSpeed := v10;
    ELSEIF speed = "v20" THEN
        moveSpeed := v20;  
    ELSEIF speed = "v30" THEN
        moveSpeed := v30;
    ELSEIF speed = "v40" THEN
        moveSpeed := v40;
    ELSEIF speed = "v50" THEN
        moveSpeed := v50;  
    ELSEIF speed = "v60" THEN
        moveSpeed := v60;
    ELSEIF speed = "v80" THEN
        moveSpeed := v80;
    ELSEIF speed = "v100" THEN
        moveSpeed := v100;  
    ELSEIF speed = "v150" THEN
        moveSpeed := v150;
    ELSEIF speed = "v200" THEN
        moveSpeed := v200;
    ELSEIF speed = "v300" THEN
        moveSpeed := v300;  
    ELSEIF speed = "v400" THEN
        moveSpeed := v400;
    ELSEIF speed = "v500" THEN
        moveSpeed := v500;
    ELSEIF speed = "v600" THEN
        moveSpeed := v600;  
    ELSEIF speed = "v800" THEN
        moveSpeed := v800;
    ELSEIF speed = "v1000" THEN
        moveSpeed := v1000;
    ELSEIF speed = "v1500" THEN
        moveSpeed := v1500;  
    ELSEIF speed = "v2000" THEN
        moveSpeed := v2000;
    ELSEIF speed = "v2500" THEN
        moveSpeed := v2500;
    ELSEIF speed = "v3000" THEN
        moveSpeed := v3000;  
    ELSEIF speed = "v4000" THEN
        moveSpeed := v4000;
    ELSEIF speed = "v5000" THEN
        moveSpeed := v5000;
    ELSEIF speed = "v6000" THEN
        moveSpeed := v6000;  
    ELSEIF speed = "v7000" THEN
        moveSpeed := v7000;
    ELSEIF speed = "vmax" THEN
        moveSpeed := vmax;
    ENDIF

ENDPROC

PROC COMH_SplitString(string fullString, INOUT string firstPart, INOUT string secondPart)
    
    VAR num deliminatorPos;
    VAR string delimnator := ";";
    
    deliminatorPos:=StrMatch(fullString,1,delimnator);
    firstPart := StrPart(fullString,1,(deliminatorPos-1));
    secondPart := StrPart(fullString, deliminatorPos+1, StrLen(fullString)-deliminatorPos);
    
    !TPWrite "full recieved " +fullString;
    !TPWrite "firstPart recieved "+firstPart;
    !TPWrite "secondPart recieved "+secondPart;
    
ENDPROC
        
ENDMODULE