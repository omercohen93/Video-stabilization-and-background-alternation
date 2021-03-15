# Video-stabilization-and-background-alternation
An algorithm that receives an unstable video of an object moving through a constant background, and peforms video stabilization, background alternation and tracking of the object.
To run the algorithm, just run the runme file. Make sure you change the initial identification of the object in Tracking (# Initial Settings)

## This algorithm is based on four stages:
### video stabilization:
In order to subtract the background, video stabilization is needed. The video was stabilized using a filter that averages the videos trajectory. The original movement of the video was estimated by using Shi-Tomasi to find interest points to track, and using them to get the homography between the images of the video. The averaging window is a moving average, with size of 150 frames. The difference in translation and rotation between average trajectory, and original trajectory was calculated for each frame, and each image was warped accordingly. 3 filters where used, where in the first tow both rotation and translation where averaged ('initial_stab' function in 'Stabilization'), and in the third only translation was taken to account, and a median was taken instead of averaging ('center_vid' function in 'Stabilization').  
### background subtraction:
Since stabilization was not ideal, background for each image was the median image of some window of images around the images, and not of all the images of the video. Because this process takes long time, the median was changed every 5 frames. The median was taken in HSV space. 3 Binary masks where created- two from S channel (one with low threshold and one with high threshold) and one from H channel.
On each of the mask some morphological actions and some color statistics where made, in order to eliminate unwanted artifacts that are not the object. 
The masks where than summed, and together with more morphological actions the final background subtraction binary mask was yielded.
### Mating:
This stage is set to better blur edges around the object for better integration of the object in the new background. We use a KDE based probability map together with Geodostic distance map. The statistics was updated every 60 frames to save runtime.
The parameters of the statistics (such as ratio R between maps, or KDE sigma) can be changed to try and improve results.
### Tracking:
Tracking of the object is done with 'Particle filter'. The algorithm needs an initial identification of the object in the first frame. 



