# CS 170 Project Spring 2020

Take a look at the project spec before you get started!

Requirements:

You'll only need to install networkx to work with the starter code. For installation instructions, follow: https://networkx.github.io/documentation/stable/install.html

Files:
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

When writing inputs/outputs:
- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
  - These are the functions run by the autograder to validate submissions

INSTRUCTIONS TO RUN CODE:
In the main section at the bottom (if __name__ == 'main' on line 428), you can adjust the start and ending inputs to be numbers from 1 to 400. Change the size string to be what size input you want to run ('small','medium','large'). Change draw_graph to be True if you want to draw the graphs as well. Then run python3 solver.py to produce the outputs, which will be stored in the output folder.

For example, if you want to run small-1.in through small-303.in, set start=1 and end=303 and size='small'.

*In addition, if you want to run the inputs on more functions, you can add more function names to the functions list on line 444.*


