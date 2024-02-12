The information in the assignment brief is designed to explain the problem that needs to be solved. While the brief contains information on what needs to be submitted, please make sure that the guidance presented here is followed as it is designed to make things work with the autograder.

Your program must contain an argument parser in the main script that allows a directory containing images to be specified. The autograder will try to run your program in the following way:

python main.py image_processing_files/xray_images/

from which it will cycle through the images in the specified directory, perform all the processing and save the images without changing the filenames in a directory called “Results”. The contents of this directory will also be a part of your submission. You program should create the Results directory if it does not already exist. 

Note that the name of the Results directory should start with a capital R. Also note that the name of the image files should not be changed before they are placed in the Results directory.

Your program should save the images in the same resolution as originally provided.

Since the point is very important, it is repeated here again. A directory called “Results” which contains the results of your image enhancement techniques on the corrupted test images should be submitted. Do not change the name, format or the resolution or the images. Note that the name of the directory “Results” should start with a capital R. Changes in the names and formats can affect the automatic grading system and will result in you losing marks. 

Make sure you do not create any extra directories or folder structures. Your submission should contain all the files that you need with the only other directory submitted being the Results directory.

Please do not include files that are not asked for.

Do not include classify.py as this will interfere with the evaluation script. The marking process will have its own version of classify.py.

As the specifications make clear "your program must contain an argument parser in the main script that allows a directory containing images to be specified." Do not hardcode image paths that might not exist when I run your code.

Note that your code needs to run with this command exactly:

python3 main.py unseen_test_imgs

I will of course provide the directory containing these unseen test images.



Do not submit your files in any extra directories. On gradescope, your submission files should be in the root directory and not in an extra directory.

For instance when you want to submit your main.py, do not submit this:

 Final_Submission/main.py

but submit this:

main.py

All files should be at the same directory level. The only directory you will submit is the "Results" directory, which should be at the same level as the rest of your files.