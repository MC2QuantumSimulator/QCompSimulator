**Gates textfile**
Some important notes in how to format the the textfile that should contain the gates that will be used in the simulator.

 - Every row should contain **ONE** gate specification
 - To gates should be formatted as an array of arrays. See example below:
 
 X = [[0,1],[1,0]] 
 T = [[0,1],[0,e^(i*p)/4]]
 
 - Whitespaces are not neccesary they will be cut out anyways.
 - The letter **i and e** will be interprested as the complex number i and the mathematical constant e, so don't use them a gatename.
 - Pi should be denoted with a p.
 - For multiplications, always use * sign.
 
 