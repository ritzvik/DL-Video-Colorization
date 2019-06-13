# Video-Colorization

### Steps to train the models

  - Make a folder `train_videos` in the root directory of the repo and put any number of `640 X 360` color videos in the directory.
  - Run `python3 scripts/frame_extract.py` to extract the color frames. It is important that the command is run from root directory.
  - Run `python3 train_scripts/1_basic.py` for training the basic model. The model would be saved in `models/1_basic.h5` after each epoch. Again, it is important for the commands to be executed from root folder.
  - Similarly run `python3 train_scripts/1_transfer.py` and `python3 train_scripts/1_transfer2.py` and run models with transfer learning.


### Steps to test 

   - Put a `640 X 360` video inside `test_videos` named `testvid1.mp4`. Ultimately, the file should be `test_videos/testvid1.mp4`.
   - To run prediction of basic model, run `python3 test_scripts/test.py models/1_basic.h5`. The predictions would be created in `models/1_basic.h5d`.
