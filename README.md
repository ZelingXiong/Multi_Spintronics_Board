# Multi_Spintronics_Board_for_Audio_Tagging
Based on the SpinTorch code in https://github.com/a-papp/SpinTorch. I have made changes to the code and changed the initial single-board system to multi-board system. Dataloader was included together with more data generation file.

Audio tag task data are avaialble on Kaggle: https://www.kaggle.com/competitions/freesound-audio-tagging-2019
With more explanation in: https://www.kaggle.com/code/maxwell110/beginner-s-guide-to-audio-data-2

Please save data in separate folder called 'input'
But remember to keep all files and folders included here, together with this 'input' folder in the same directory
Also, remember to change your_path on line 20 in each file to your directory path. 
By running the Audio_Single_Board.py file, single-spintronics board training will begin.
By running the Audio_Double_Board.py file, double-spintronics board training will begin.
