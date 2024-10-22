# amc-oven

# Set-up for the code:
The number of temperature updates, the minimum and maximum values of the temperature changes, and the bounds of the temperatures need to be edited in opt_step.py. The root directory can also be changed there.

# Running the code:

1. Run opt_step.py, this will create a neural network protocol (now as a txt file, I can change this to some other format if convenient)
2. Write the order parameter you are trying to maximize to a file called "results.json" (can also change this if wanted) in the directory corresponding to the latest experiment. This will be in root+"batch"+str(n) where n is the number of steps you've taken.
3. Repeat
