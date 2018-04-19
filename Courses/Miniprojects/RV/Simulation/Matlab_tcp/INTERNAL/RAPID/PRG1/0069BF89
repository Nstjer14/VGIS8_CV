MODULE FunctionHandler 
    
PERS tooldata defaultTool;
PERS wobjdata defaultWobj;

VAR signaldi defaultGripperActionExecutedIO;
VAR signaldi defaultGripperObjectStateIO;
VAR signaldo defaultGripperActivationIO;

VAR signaldi defaultTakePictureExecutedIO;
VAR signaldo defaultTakePictureActivationIO;

PROC FH_GetPosition(INOUT pos currentPos, INOUT orient currentOrient)
    
    VAR robtarget tempTarget;

    tempTarget := CRobT(\Tool:=defaultTool,\Wobj:=defaultWobj);
    currentPos := tempTarget.trans;
    currentOrient := tempTarget.rot;
    
ENDPROC

PROC FH_GetJointValues(INOUT jointtarget jointValues)

    jointValues := CJointT();
    
ENDPROC

PROC FH_GripperActivation(num activate, INOUT num objectGrasped)
    
    SetDO defaultGripperActivationIO,activate;
    WaitDI defaultGripperActionExecutedIO,1;
    WaitTime 0.1;
    objectGrasped := defaultGripperObjectStateIO;
    
ENDPROC

PROC FH_MoveJoints(jointtarget jointValues, speeddata moveSpeed)
    
    VAR jointtarget temptarget;
    
    temptarget := CJointT();
    temptarget.robax := jointValues.robax;
    MoveAbsJ temptarget, moveSpeed, fine, defaultTool; 
    WaitUntil\InPos, TRUE;
    
ENDPROC

PROC FH_MoveLinear(pos movePos, orient moveOrient, speeddata moveSpeed)
    
    VAR robtarget tempTarget;

    tempTarget := CRobT(\Tool:=defaultTool,\Wobj:=defaultWobj);
    tempTarget.trans.x := movePos.x;
    tempTarget.trans.y := movePos.y;
    tempTarget.trans.z := movePos.z;
    tempTarget.rot.q1 := moveOrient.q1;
    tempTarget.rot.q2 := moveOrient.q2;
    tempTarget.rot.q3 := moveOrient.q3;
    tempTarget.rot.q4 := moveOrient.q4;

    MoveL tempTarget, moveSpeed, fine, defaultTool \Wobj:=defaultWobj;
    WaitUntil\InPos, TRUE;
    
ENDPROC

PROC FH_MovePTP(pos movePos, orient moveOrient, speeddata moveSpeed)
    
    VAR robtarget tempTarget;

    tempTarget := CRobT(\Tool:=defaultTool,\Wobj:=defaultWobj);
    tempTarget.trans.x := movePos.x;
    tempTarget.trans.y := movePos.y;
    tempTarget.trans.z := movePos.z;
    tempTarget.rot.q1 := moveOrient.q1;
    tempTarget.rot.q2 := moveOrient.q2;
    tempTarget.rot.q3 := moveOrient.q3;
    tempTarget.rot.q4 := moveOrient.q4;

    MoveJ tempTarget,moveSpeed,fine,defaultTool \Wobj:=defaultWobj;
    WaitUntil\InPos, TRUE;
    
ENDPROC

PROC FH_MoveRelativeLinear(pos movePos, orient moveOrient, speeddata moveSpeed)
    
    VAR robtarget tempTarget;

    tempTarget := CRobT(\Tool:=defaultTool,\Wobj:=defaultWobj);
    tempTarget.trans.x := tempTarget.trans.x + movePos.x;
    tempTarget.trans.y := tempTarget.trans.y + movePos.y;
    tempTarget.trans.z := tempTarget.trans.z + movePos.z;
    tempTarget.rot.q1 := tempTarget.rot.q1 + moveOrient.q1;
    tempTarget.rot.q2 := tempTarget.rot.q2 + moveOrient.q2;
    tempTarget.rot.q3 := tempTarget.rot.q3 + moveOrient.q3;
    tempTarget.rot.q4 := tempTarget.rot.q4 + moveOrient.q4;

    FH_MoveLinear tempTarget.trans,temptarget.rot, moveSpeed;
    
ENDPROC

PROC FH_MoveRelativePTP(pos movePos, orient moveOrient, speeddata moveSpeed)
    
    VAR robtarget tempTarget;

    tempTarget := CRobT(\Tool:=defaultTool,\Wobj:=defaultWobj);
    tempTarget.trans.x := tempTarget.trans.x + movePos.x;
    tempTarget.trans.y := tempTarget.trans.y + movePos.y;
    tempTarget.trans.z := tempTarget.trans.z + movePos.z;
    tempTarget.rot.q1 := tempTarget.rot.q1 + moveOrient.q1;
    tempTarget.rot.q2 := tempTarget.rot.q2 + moveOrient.q2;
    tempTarget.rot.q3 := tempTarget.rot.q3 + moveOrient.q3;
    tempTarget.rot.q4 := tempTarget.rot.q4 + moveOrient.q4;

    FH_MovePTP tempTarget.trans, temptarget.rot, moveSpeed;
    
ENDPROC

PROC FH_TakePicture()
    
    SetDO defaultTakePictureActivationIO,1;
    WaitDI defaultTakePictureExecutedIO,1;
    SetDO defaultTakePictureActivationIO,0;
    
ENDPROC

ENDMODULE