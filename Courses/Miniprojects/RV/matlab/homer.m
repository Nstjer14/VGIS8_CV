%% Make Homer
%
% brick order
% blue - red - yellow


    disp('Building Homer');
    brickPos = Bricks{1,4}{YL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,4}{YL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);
    YL = YL-1;

    disp('Yellow brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*2;



    brickPos = Bricks{1,1}{RL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,1}{RL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);
    RL = RL-1;

    disp('Red brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*1.5;

    brickPos = Bricks{1,3}{BL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,3}{BL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);
    BL = BL-1;

    disp('Blue brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight/3;