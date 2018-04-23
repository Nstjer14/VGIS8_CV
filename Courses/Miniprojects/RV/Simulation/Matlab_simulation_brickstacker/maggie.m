%% Make Maggie
%
% brick order
% yellow - red


    disp('Building Maggie');
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

    brickCol = 1; % Orange

    brickPos = Bricks{1,brickCol}{RL}.boudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,brickCol}{RL}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);
    RL = RL-1;

    disp('Orange brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight/3;