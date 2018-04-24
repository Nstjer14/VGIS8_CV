%% Make Marge
%
% brick order
% green - yellow - blue


    disp('Building Marge');
    brickCol = 3; % Blue
    brickPos = Bricks{1,brickCol}{BL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,brickCol}{BL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    myConnector.movePTP(BrickPoseUR,'v100');;
    BL = BL-1;

    disp('Blue brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*2;

    brickCol = 4; % Yellow

    brickPos = Bricks{1,brickCol}{YL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,brickCol}{YL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    myConnector.movePTP(BrickPoseUR,'v100');;
    YL = YL-1;

    disp('Yellow brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*1.5;
    
    brickCol = 2; % Green
    brickPos = Bricks{1,brickCol}{GL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,brickCol}{GL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    myConnector.movePTP(BrickPoseUR,'v100');;
    GL = GL-1;
    
    disp('Green brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight/3;