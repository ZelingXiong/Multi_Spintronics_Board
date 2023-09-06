# Multi_Spintronics_Board_for_Audio_Tagging
Based on the SpinTorch code in https://github.com/a-papp/SpinTorch. 

I have made changes to the code and changed the initial single-board system to multi-board system. Dataloader was included together with more data generation file.

### Data
Audio tag task data are avaialble on Kaggle: https://www.kaggle.com/competitions/freesound-audio-tagging-2019

With more explanation in: https://www.kaggle.com/code/maxwell110/beginner-s-guide-to-audio-data-2

The data.py file can be used to generate simpler multi-freuqnecy RF signals in sinusoidal wave forms.

### How to Run the simulation
Please save data in separate folder called 'input'

But remember to keep all files and folders included here, together with this 'input' folder in the same directory

Also, remember to change your_path on line 20 in each file to your directory path. 

1) By running the Audio_Single_Board.py file, single-spintronics board training for audio tagging task will begin.

2) By running the Audio_Double_Board.py file, double-spintronics board training for audio tagging task will begin.

3) By runnning the RF_single_board.py, single-spintronics board training for multi-frequency RF signal classification task will begin.

4) By runnning the RF_double_board.py, double-spintronics board training for multi-frequency RF signal classification task will begin.

Np in each file represents number of different signals included in the classification task.

Suitable range for Np is 2 to 10.


### NOTE
The training is very time and energy consuming, make sure you have enough time and connected to power:)
