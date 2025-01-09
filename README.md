## An improved differential evolution algorithm and its application in optimization problem

This repository is related to the implementation of this [article](https://link.springer.com/article/10.1007/s00500-020-05527-x).\
In this project I implemented a modified differential evolution algorithm featuring a unique mutation vector and a new operator called Opposition-Based Learning
addressing exploration ability, convergence accuracy and convergence speed.

The new mutation vector contains X_best(the best individual), X_3(one randomly chosen individual of top 30%) and two scaling factors F1 and F2: 

![image](https://github.com/user-attachments/assets/9a676caf-ed2c-493c-823b-d4bd638ec992)

The main idea of OBL is to evaluate the fitness values of the current point and its
reverse point in order to correct the convergence direction and select the better solutions.

![image](https://github.com/user-attachments/assets/e7834567-3c22-4d70-a602-20dfd8f79f55)

Finally, the proposed algorithm is evaluated by 12 benchmark functions with low-dimension and high-dimension.
