%% Make Lisa
%
% brick order
% yellow - orange - yellow


    disp('Building Lisa');
    brickCol = 4; % Yellow
    brickPos = Bricks{1,brickCol}{YL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,brickCol}{YL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);
    YL = YL-1;

    disp('Yellow brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*2;

    brickCol = 5; % Orange

    brickPos = Bricks{1,brickCol}{OL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,brickCol}{OL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);
    OL = OL-1;

    disp('Orange brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*1.5;
    
    brickCol = 4; % Yellow
    brickPos = Bricks{1,brickCol}{YL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,brickCol}{YL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);
    YL = YL-1;
    
    disp('Yellow brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight/3;