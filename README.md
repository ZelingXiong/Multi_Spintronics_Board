# Multi_Spintronics_Board
Based on the SpinTorch code in https://github.com/a-papp/SpinTorch. 

I have made changes to the code and changed the initial single-board system to multi-board system. Dataloader was included together with more data generation file.

### Data
Audio tag task data are avaialble on Kaggle: https://www.kaggle.com/competitions/freesound-audio-tagging-2019

<img src= "![original_plot](https://github.com/ZelingXiong/Multi_Spintronics_Board/assets/92733114/d69c90e2-ec17-4e1b-9246-f5b0cfcd0671)" width="300" height="200">

![Final_Plot](https://github.com/ZelingXiong/Multi_Spintronics_Board/assets/92733114/2a6fab37-d619-451d-bde3-73079d9c9c16)

With more explanation in: https://www.kaggle.com/code/maxwell110/beginner-s-guide-to-audio-data-2

Multi-freuqnecy RF signals are generated using data.py, they are simpler than audio data.
![ABABAsource_signal](https://github.com/ZelingXiong/Multi_Spintronics_Board/assets/92733114/2f2e4dc1-5b41-47eb-a458-e16e03d6c7bd)
![ABABsource_FOURIER](https://github.com/ZelingXiong/Multi_Spintronics_Board/assets/92733114/2ac446a0-1a94-45ae-9d1e-366e00d2cb5b)

### How to Run the simulation
Please save audio data in separate folder called 'input'

But remember to keep all files and folders included here, together with this 'input' folder in the same directory

Also, remember to change your_path on line 20 in each file to your directory path. 

1) By running the Audio_Single_Board.py file, single-spintronics board training for audio tagging task will begin.

2) By running the Audio_Double_Board.py file, double-spintronics board training for audio tagging task will begin.

3) By runnning the RF_single_board.py, single-spintronics board training for multi-frequency RF signal classification task will begin.

4) By runnning the RF_double_board.py, double-spintronics board training for multi-frequency RF signal classification task will begin.

Np in each file represents number of different signals included in the classification task.

Suitable range for Np is 2 to 10.
![snapshot_time1100_amplitude 1 00X1](https://github.com/ZelingXiong/Multi_Spintronics_Board/assets/92733114/df24cc1b-74cd-418d-b661-7abcc823d815)


### NOTE
The training is very time and energy consuming, make sure you have enough time and connected to power:)
