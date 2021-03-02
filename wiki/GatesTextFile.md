**Gates textfile**
Some important notes in how to format the the textfile that should contain the gates that will be used in the simulator.

 - Every row should contain **ONE** gate specification
 - To gates should be formatted with ';' in between every row, and ',' in between every element. See example below:
 
X = 1,2;3,1/(2^(1/2)) = 2
P = 1,0;1,1 = 2
 
 - Whitespaces are not neccesary they will be cut out anyways.
 - The number after the last '=' sign stand for the dimension of the matrix. If the matrix is 2x2, put a 2 there.
 - **To be added: How to input math format**
 
 ---------------------------------------------------------------------
 
  - separera element med ',' och rader med ';' ALternativt python syntax
  - Addera global vikt på något sätt, för att inte behöva skriva ut på varje element
  - Göra om Gates_parser ouput till numpy matris
 
 