#CSC323 Lecture 3
2014-01-21

## Images, 3D Plots, and Derivatives
- Images are just a mapping from 2 variables `(x,y)` to a third `(color value)`
- Same as a 3 axis plot `(x,y) -> (z)`
- `np.meshgrid`


### Taking derivatives of 2 axis plots
- partial derivative 
 	$\frac{\delta}{\delta x} f(x,y) = \frac{d}{dx} f_y(x)$
 	$\frac{\delta}{\delta y} f(x,y) = \frac{d}{dy} f_x(y)$
- fix one of the variables, looking at a slice of the function along one axis.
- the full derivative of an n variable function is the Gradient (in our case never more than 2)
 	$$$
	\nabla f =
    \left[ \begin{array}{c}
    \frac{\delta f}{\delta x} \\
  	\frac{\delta f}{\delta y}
 	\end{array} \right]
    $$$
- magnitude $|| \nabla f ||$
- approximate by taking the secant at intervals along each axis
    $\approx \frac{f(x+a)-f(x)}{a}$

## Edges
- What are edges?
    - naive approach: Discontinuities in the intensity of an image
    - naive approach leads to issues
- What are edges, really?
    - Surface normal discontinuity (sharp changes in the surfaces of objects)
    - Depth discontinuity
    - Surface color discontinuity
    - illumination discontinuity
- soft edges (hair, fur, etc) make continuous objects blurry


### Characterizing Edges
- rapid change in the image intensity function
- Small amounts of noise appear large in the gradient
    - Solution: small gaussian filter applied before calculating the gradient


### Derivative Theorem of COnvolution
- Differentation is convolution and convoluion is associative
$ \frac{d}{dx}(f \star g) = f \star \frac{d}{dx} g $

- same g is applied for all points along f, so calculating the derivative of g is a simple but very effective optimization step.

### Terminology
- **Edge Normal**: unit vector in the direction of maximum intensity change
- **Edge Direction**: unit vector perpendicular to edge normal
- **Edge Position/Center**: image position at which the edge is located
- **Edge Strength/Magnitude**: local image contrast along the normal.


