# Multi_Spintronics_Board
Based on the SpinTorch code in https://github.com/a-papp/SpinTorch. 

I have made changes to the code and changed the initial single-board system to multi-board system. Dataloader was included together with more data generation file.

## Data
### Audio Data
Audio tag task data are avaialble on Kaggle: 

https://www.kaggle.com/competitions/freesound-audio-tagging-2019

<img src="https://github.com/ZelingXiong/Multi_Spintronics_Board/blob/main/Pictures/original_plot.png" width="400" height="250">
<img src="https://github.com/ZelingXiong/Multi_Spintronics_Board/blob/main/Pictures/Final_Plot.png" width="400" height="250">

With more explanation in: https://www.kaggle.com/code/maxwell110/beginner-s-guide-to-audio-data-2

### Multi-freuqnecy RF signals
Multi-freuqnecy RF signals are generated using data.py, they are simpler than audio data.
<img src="https://github.com/ZelingXiong/Multi_Spintronics_Board/blob/main/Pictures/ABABAsource_signal.png" width="400" height="370">
<img src="https://github.com/ZelingXiong/Multi_Spintronics_Board/blob/main/Pictures/ABABsource_FOURIER.png" width="400" height="370">


## How to Run the simulation
Please save audio data in separate folder called 'input'

But remember to keep all files and folders included here, together with this 'input' folder in the same directory

Also, remember to change your_path on line 20 in each file to your directory path. 

1) By running the Audio_Single_Board.py file, single-spintronics board training for audio tagging task will begin.

2) By running the Audio_Double_Board.py file, double-spintronics board training for audio tagging task will begin.

3) By runnning the RF_single_board.py, single-spintronics board training for multi-frequency RF signal classification task will begin.

4) By runnning the RF_double_board.py, double-spintronics board training for multi-frequency RF signal classification task will begin.

Np in each file represents number of different signals included in the classification task.

Suitable range for Np is 2 to 10.

<img src="https://github.com/ZelingXiong/Multi_Spintronics_Board/blob/main/Pictures/snapshot_time1100_amplitude%203.00X1.png" width="600" height="250">


## NOTE
The training is very time and energy consuming, make sure you have enough time and connected to power:)
