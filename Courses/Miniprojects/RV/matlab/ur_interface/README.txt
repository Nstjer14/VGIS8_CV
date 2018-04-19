%%%%%% Description %%%%%%
This package enables communication between UR5 and matlab over TCP. A .urp
file is executed on the UR controller (polyscope) and a set of matlab files
then enables communication. There are a number of .urp files provided for
different versions of polyscope; named "matlab_rsa_x.x.urp", where x.x is the
version no.. The matlab file "demo.m" shows how to use most of the
functionality.

%%%%%% GUIDE %%%%%%
1) Copy "matlab_rsa_x.x.urp" to the UR robot (using USB)
2) Open demo.m on the PC and "matlab_rsa_x.x.urp" on the UR controller
3) IP adddresses
   a) Setup static IP on the PC and try to ping the UR controller
   b) Correct 'robot_ip' in the matlab script 
   c) Correct the IP on the UR controller in the line 'socket_open(...)'
4) Start demo.m
5) When it says 'Press Play on robot', do it
   The robot will now move, so BE READY WITH THE EMERGENCY STOP!

%%%%%% Versions %%%%%%
This package has been tested with:
 - matlab_rsa_3.4.urp
      x ursim 3.4 and 3.5
      x UR5 3.3.4.310
      x UR5 3.5.0.10584
 - matlab_rsa_3.3.urp
      x (not tested)
 - matlab_rsa_3.0.urp
      x UR5 3.0.???
 - last_test_1.8.urp
      x (not tested)
