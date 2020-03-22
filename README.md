# DAC-DC : Divide and Concquer for Detection and Classification

![](media/dac-dc.gif)

This is a modified version of YOLO for performing 2D object detection and tracking. The model was trained and tested on several datasets and seems to be performing quite well. The results shown here are for virtualKITTI dataset across multiple weather conditions and camera positions.

![](media/result.png)
*Figure 1: Result of DAC-DC on virtualKITTI dataset. Blue boxes are ground-truth, and red boxes are predictions*

## Train/Test DAC-DC on virtualKITTI dataset

 - Download the dataset. 
    ```
    wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar
    wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz
    ```

 - Extract the dataset. 
    ```
    tar -xvf vkitti_2.0.3_rgb.tar
    tar -xvf vkitti_2.0.3_textgt.tar.gz
    ```

 - Download the pretrained weight. 
   ```
   wget https://drive.google.com/file/d/176AQBw9NkcH2vY9b4OsidcvjYwKQIhd4/view?usp=sharing
   ```

 - Modify the data path of dataset in ```config/config.ini``` by changing the line ```DATA_PATH = ``` 

 - Modify the pretrained weight file path in ```config/config.ini``` by changing the line ```PRETRAINED_WEIGHT_FILE = ``` 

 - Train the network. 
    ```
    python trainModel.py
    ```

 - Train the network. 
    ```
    python testModel.py
    ```