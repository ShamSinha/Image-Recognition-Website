# Image Recognition Website


## In Windows 
1. [Install Anaconda ](https://docs.anaconda.com/anaconda/install/)
        
2. Now we install all the packages required to run our website using Anaconda prompt command line
   - Install OpenCV library
   
         conda install -c conda-forge opencv
   - Install Tensorflow-cpu  
   
         conda install tensorflow-cpu
   - Open SPYDER 
         
         spyder
 
         
3. Now clone this github master branch repo in your local machine.
4. Run app.py in spyder 


## Landmarks Recognition Model


**These are the 30 landmarks that our Landmark recognition model written in Keras can recognise with 77.55% accuracy in validation set which comprise of 590 images and training set comprise of 5300 images.**

[Install Keras model](https://drive.google.com/file/d/1VtiL6fqkCT_ehzRP2lBPo71hqL-wyC5c/view?usp=sharing)

Procedures to load Landmark dataset in Google Colab used for Training and Testing of our **Landmark recognition Model**

1. Open this [link](https://drive.google.com/file/d/1VtiL6fqkCT_ehzRP2lBPo71hqL-wyC5c/view?usp=sharing) in your Google Drive 
2. Open Shared with me within Google Drive and use Add Shortcut to Drive option to make shortcut of Landmark_data30.npz dataset file in My Drive
3. Open google Colab and run
        ```python
        
        from google.colab import drive
        drive.mount('/content/drive')
        ```
4. After successful mount the drive run this to load dataset
        ```python
        
        npzfile = np.load('/content/drive/My Drive/Landmark_data30.npz')
        
        X = npzfile['arr_0']
        
        Y = npzfile['arr_1']
        
        np.random.seed(150)
        
        np.random.shuffle(X)
        
        np.random.seed(150)
        
        np.random.shuffle(Y)
        ```

1. Angkor Wat
2. Arc de Triomphe
3. Big Ben
4. Burj Al Arab
5. Burj Khalifa
6. Charminar
7. Chichen Itza
8. Christ The Redeemer
9. Colosseum
10. Effiel tower
11. Empire state building
12. Golden Gate Bridge
13. Golden Temple
14. Hawa Mahal
15. India Gate
16. Leaning Tower of Pisa
17. Lotus temple
18. Mount Rushmore
19. Petronas Tower
20. Pyramid of Giza
21. Qutub Minar
22. Red Fort
23. Statue of Liberty
24. Stone Henge
25. Sydney opera house
26. Taj Hotel
27. Taj Mahal
28. Tower Bridge
29. Victoria Memorial
30. White house



![](https://github.com/ShamSinha/Image-Recognition-Website/blob/branch1/images/1.png?raw=true)
![](https://github.com/ShamSinha/Image-Recognition-Website/blob/branch1/images/2.png?raw=true)


