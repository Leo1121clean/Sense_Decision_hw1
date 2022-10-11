Task1: BEV projection
1. run load.py
2. press 'w','a' or 'd' to move in the apartment
3. save fornt_view image and BEV image by pressing 'q' and 'e' respectively, and press 'f' to quit
4. run bev.py
5. choose an area and click points to surround it on the BEV image, and press enter.
6. the projection of this area is on the front_view image.

Task2: ICP Alignment and Reconstruction
1. run load.py
2. press 'w','a' or 'd' to move in the apartment, and walk through all the areas
3. run reconstruct.py, and wait for a few minutes
4.(1) the 3D environment of the apartment, estimated trajectory as well as ground truth trajectory are all showed in open3d
4.(2) the mean distance between estimated trajectory andground truth trajectory is showed on the terminal
