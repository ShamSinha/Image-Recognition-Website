# Image Recognition Website


## In Windows 
1. [Install Anaconda ](https://docs.anaconda.com/anaconda/install/)
2. Open Anaconda Prompt to create a new environment name myenv where you install every packages required to run this site.

        conda create --name myenv python=3.5
3. Activate myenv environment

        conda activate myenv
        
       
        
**To Install all the packages we need to run this website.**


## Landmarks Recognition Model


**These are the 30 landmarks that our Landmark recognition model written in Keras can recognise with 70% accuracy in validation set which comprise of 590 images and training set comprise of 5300 images.**

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

**If we want to increase the number of classes in the model then we just have to remove the last Softmax layer of current model to new Softmax layer**
