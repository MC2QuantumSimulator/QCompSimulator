# **Gates textfile**

Some important notes in how to format the the textfile that should contain the gates that will be used in the simulator.

 - Every row should contain **ONE** gate specification
 - Every gate should be encapsulated by hard brackets '[gate]'
 - Global weights should be added before gate definition and '*' afterwards to indicate a global factor.
 - To gates should be formatted with ';' in between every row, and ',' in between every element. See example below:

#### Examples of common gates:
 
H = 1/(2^(1/2))*[1,1;1,-1]

Z = [1,0;0,-1]

CX = [1,0,0,0;0,1,0,0;0,0,0,1;0,0,1,0]

S = [1,0;0,i]

T = [1,0;0, e^(i*PI/4)
 
 - Whitespaces are not neccesary they will be cut out anyways.
 
 ---------------------------------------------------------------------
 
 
 