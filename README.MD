# DOUBLE DEEP Q-LEARNING CHESS AGENT
Our github repo is linked below: (It may be updated as this was a fun project to make, but the  zipped files contains everything you need to run ui.py)
https://github.com/woodphrog/andoma

## Play the agent
	* Go to ui.py file 
	* Select which model from checkpoints you would like to run
		- depending on your OS you will need to change the path in dql.load
		- make sure to use the same modelname for both model and target model (i.e. use *model*.h5 and *targetModel*.h5 where * match)
	* Run the file and play though the command line (following the instructions given)
		- You may be required to install some dependencies (tensorflow, chess, etc)
		- ENJOY !! (and let me know  your results)

## Train the Agent
	* Run agent.py with your desired hyperparameters and PGN dataset
	* I would not recommend as it will take several hours to train 

## Test the agent
	* Run generateGamesStockfish.py
		- This will require you to have a working stockfisg engine on your computer
	* Again would not recommend due to set-up time and runtime for the tests (A few hrs per model)

## Sidenotes 
	* Since the agent takes a long time to train the /checkpoints directory is used to save the weights of our trained models
	* Please contact me at thomas.inman@mail.mcgill.ca if you have any questions, comments, or concerns
	* I made sure to test by clonning the repository so it should work on most platforms, bbut if you encounter issues let me know