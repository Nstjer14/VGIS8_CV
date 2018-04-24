%% Make Homer
%
% brick order
% blue - red - yellow


    disp('Building Homer');
    
    
    brickPos = Bricks{1,3}{BL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+(BrickHeight*0.5)];
    
    myConnector.movePTP([worldBrickPos 0 0 1 0],'v100');
    J_val = myConnector.getJointValues();
    J_val(6) = J_val(6)-Bricks{1,3}{BL}.rotation;
    myConnector.moveJoints(J_val,'v100');
    %currentPos = myConnector.getPosition();
    %Quart = currentPos(4:end);
    %OrientInDeg = quat2eul(Quart);
    %OrientRotated = OrientInDeg*rotz(-Bricks{1,4}{BL}.rotation);
    %newQuart = eul2quat(OrientRotated);
    %myConnector.movePTP([currentPos(1:3) newQuart],'v100');
    
    pause();
    BL = BL-1;

    disp('Blue brick has been picked up');
    myConnector.movePTP(homepos,'v100');
    myConnector.movePTP(dropOffPos,'v100');
    pause()
    
    
    %myConnector.gripperOn();
    %%
    %BrickHeight = BrickHeight*1.5;
    dropOffPos(3) = dropOffPos(3)+ BrickHeight;


    brickPos = Bricks{1,1}{RL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+(BrickHeight*0.5)];
    
    myConnector.movePTP([worldBrickPos 0 0 1 0],'v100');
    J_val = myConnector.getJointValues();
    J_val(6) = J_val(6)-Bricks{1,1}{RL}.rotation;
    %J_val(6) = J_val(6)-(-0.280825678518857)
    myConnector.moveJoints(J_val,'v100');
    pause();
    RL = RL-1;

    disp('Red brick has been picked up');


    myConnector.movePTP(homepos,'v100');
    myConnector.movePTP(dropOffPos,'v100');
    pause()
    %%
    %BrickHeight = BrickHeight*1.5;
    dropOffPos(3) = dropOffPos(3)+ BrickHeight;
    brickPos = Bricks{1,4}{YL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+(BrickHeight*0.5)];
    
    myConnector.movePTP([worldBrickPos 0 0 1 0],'v100');
    J_val = myConnector.getJointValues();
    J_val(6) = J_val(6)+Bricks{1,4}{YL}.rotation;
    myConnector.moveJoints(J_val,'v100');
    pause();

    YL = YL-1;
    disp('Yellow brick has been picked up');

    myConnector.movePTP(homepos,'v100');
    myConnector.movePTP(dropOffPos,'v100');
    pause()