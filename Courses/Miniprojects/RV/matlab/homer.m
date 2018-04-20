%% Make Homer
%
% brick order
% blue - red - yellow


    disp('Building Homer');
    brickPos = Bricks{4,1}{length(Bricks{4,1})}.BoudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{4,1}{length(Bricks{4,1})}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);

    disp('Blue brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*2;



    brickPos = Bricks{1,1}{length(Bricks{1,1})}.BoudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{1,1}{length(Bricks{1,1})}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);

    disp('Red brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight*1.5;

    brickPos = Bricks{3,1}{length(Bricks{3,1})}.BoudingBoxCenter;
    worldBrickPos = [[1 brickPos] * [theta phi] zHeight+BrickHeight];

    ToolPose = GetSO4FromURpose(homepos);
    BrickPose = ToolPose * trotz(deg2rad(-Bricks{3,1}{length(Bricks{3,1})}.rotation));
    BrickPose(1:3,4) = worldBrickPos;
    BrickPoseUR = GetURposeFromSO4(BrickPose);

    urMoveL(sock, BrickPoseUR);

    disp('Yellow brick has been picked up');


    urMoveL(sock,homepos);
    BrickHeight = BrickHeight/3;