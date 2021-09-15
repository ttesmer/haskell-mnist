# Notes
- NN is just a Universal Function Approximator
### Backpropagation
### Quadratic Cost Function
- Squared Error: |y2-y1|^2
- Mean Squared Error: 1/2n * sum_x (y(x) - a)^2
- Advantages of quadratic cost function: 
    - Undershoot and Overshoot both have same consequences because of symmetry
    - The constant C in C(y-a)^2 can be adjusted to anything because it doesn't influence the updates to the weights and biases
        - Thus, it can be changed to eg 1/2 so that for the derivative it cancels out with the power of 2 which jumps in front.
        - Then it's easier for gradient descent because it's just y-a. (? u sure?) (sorta, 1/2n becomes just n so theres still something left, but it's simpler.)
### Gradient descent
- It's just the first order derivative of cost function!

### IMPORTANT
- perhaps 0-255 unsigned 8-bit integers arent to be normalized by n(x)=x/255 but instead converted to *actual numbers*?
