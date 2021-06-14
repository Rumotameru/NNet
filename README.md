# NNet
## Identify cancer cells in the most prevalent childhood cancer type.
##### My version of the solution to [kaggle competition](https://www.kaggle.com/andrewmvd/leukemia-classification).

    python 3.7.4
    pandas 0.25.1
    sklearn 0.21.3
    torch 1.8.1+cu111	
    matplotlib 3.1.1	

Training of the model performs in NN_train.py. The final state of the model is saved in model_state.pt. 

The main program (with GUI) that use the pre-trained model is located in design.py 

1. Run design.py.
2. Press "Browse" button and select pre-processed  image of the cell from "Images for program's testing" folder. Also you can specify the path to image manually.
3. Press "Check" button.
4. The selected cell image with the predicted label will be displayed in a specially designated field next to "Check" button.

![image](https://github.com/Rumotameru/NNet/blob/master/Info.png?raw=true)
