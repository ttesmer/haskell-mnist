# Notes
- Neural Network = Universal Function Approximator
### Backpropagation
### Gradient Descent
### Quadratic Cost Function
- Squared Error: |y2-y1|^2
- Mean Squared Error: 1/2n * sum_x (y(x) - a)^2
- Advantages of quadratic cost function: 
    - Undershoot and Overshoot both have same consequences because of symmetry
    - The constant C in C(y-a)^2 can be adjusted to anything because it doesn't influence the updates to the weights and biases
        - Thus, it can be changed to eg 1/2 so that for the derivative it cancels out with the power of 2 which jumps in front.
