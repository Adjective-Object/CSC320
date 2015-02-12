# CSC320 Lecture 5
faces and face spaces

# Faces and Face Spaces
Look at the set of n-dimensional vectors that makes up all images of a given size and shape, There exists some subset of those vectors which are "images of faces"

Each face image is a point in n-dimensional space.
We take a large sample of these faces, and attempt to find a the basis of the subspace that approximates that set of points.

This is the so-called "Face-Space"

Because subspaces are always centred about the origin, we offset the entire space by some set amount (usually the average of the sample face points).

### Change of basis / Projection
a basis:
- a new basis $(V_1, V_2)$
- $Projection_X = (X \dot V_1) V_1 + (X \dot V_2) V_2$
-	 THIS IS IMPORTANT

In 2 dimensions, the projection of a point upon a line is the closest point on that line to the point. 
Therefore, the distance between the projection of the point onto the line and the point itself is the shortest distance from the point to that line. 

In order to determine if a point is a face, we set some threshold as "close enough to the face-space to be considered a face"

This same principle can be applied along N dimensions

## Face Detection using PCA, Formalized
- For each (centered) window x and for a set of principal components V, compute the Euclidian distance 
$|VV^T x - x|$
	- That is the distance betwen the reconstuction of c and c. The reconstruction of x is similar to x if x lies in the face subspace.
	- The reconstruction of x is the closest space in the face subspace to x

- Win: instead of comparing x to a large dataset of faces, we are only comparing x to the columns of V
	- $V^T_x$ is just a vector of the dot products 
		$v_i \dot x$ for every I
	- That still works, since V contains (we hope) all the information about the appearance of the face

## Nonlinear Methods
Not all spaces can be accurately summarized by a linear approximation. However, these same basic techniques (trendline + difference) apply to nonlinear trends.

We're not going into detail on those

# A LITTLE BIT OF MATH

## POLAR COORDINATES
you sure do remember this from highschool
$(a,b)$ in cartesian coordinates is 
$(t, r)$
where $a = r cos(t), b= r sin(t)$

## IMAGINARY NUMBERS AND POLAR COORDINATES
$r(cos(t) + i sin(t)) = r e^{i t} = e^{log r + i t}$  
Where t is $\theta$

# Fourier Transforms
(almost) Any univariate function can be written as a weighted sum of sines and cosines at different frequencies

This is called the Fourier series for a function. It's usually an infinite series.

The "coarse" structure is encoded with long wavelength sine waves, the "fine" structure is encoded with short wavelength waves.

Note: $F(w) = F(\omega)$

$f(x) - Fourier Transform -> F(w)$  
where $f(x) = \sum_{w=0}^{\inf}{A sin(wx + \phi)}$  
and $f(w)$ holds the amplitude $A$ and phase $\phi$ of the corresponding sine.
where $F(w) = R(w) +iI(w)$
$A = += \sqrt{R(w)^2 + I(w)^2}$, $\phi = tan^-1 \frac{I(w)}{R(w)}$

Usually, Frequency is more interesting than Phase

### Frequency Spectra
This is what you usually think of when you think of the Fourier Transform

### Fourier Transform as a Change of Basis
![Fourier-Transform](https://i.imgur.com/s3xiOUf.png)

## Discrete 2D Fourier Transform

