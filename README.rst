Magic Cube
==========

It had to happen someday.  Somebody stop me!

.. image:: http://4.bp.blogspot.com/-iruqaXDstKk/UKBejowDVkI/AAAAAAAAZkM/c2tir0qcexQ/s400/test04.png
   :alt: cube views
   :align: left

Authors
-------

- **David W. Hogg** (NYU)
- **Jacob Vanderplas** (UW)

Usage
-----

Interactive Cube
~~~~~~~~~~~~~~~~
To use the matplotlib-based interactive cube, run 

     python code/cube_interactive.py
This script also saves the images to code/images, and the corresponding Quaternion Rotations of the Cube to code/quaternions.csv

If you want a cube with a different number of sides, use e.g.

     python code/cube_interactive.py 5

This will create a 5x5x5 cube

This code should currently be considered to be in beta --
there are several bugs and the GUI has an incomplete set of features

Controls
********
- **Click and drag** to change the viewing angle of the cube.  Holding shift
  while clicking and dragging adjusts the line-of-sight rotation.
- **Arrow keys** may also be used to change the viewing angle.  The shift
  key has the same effect
- **U/D/L/R/B/F** keys rotate the faces a quarter turn clockwise.  Hold the
  shift key to rotate counter-clockwise.  Hold a number i to turn the slab
  at a depth i (e.g. for a 3x3 cube, holding "1" and pressing "L" will turn
  the center slab).

Other - Continued by Srinath (hi there! :D)
~~~~~
I've modified this visualization software to generate a dataset for estimating Quaternion Rotation Matrices using images of a Rubik's Cube in different orientations.

AICrowd had a similar dataset, but it was plagued by differences in illumination, which made it extremely difficult to work with. MagicCube does not have any of those problems!



This problem can be solved using Machine Learning and Image Regression, but I'm trying out Image Geometry and Simple Linear Algebra for this! - Stay tuned for more updates.


License
-------

All content copyright 2012 the authors.
**Magic Cube** is licensed under the *GPLv2 License*.
See the `LICENSE.txt` file for more information.
