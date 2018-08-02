## TTS example web-server
Steps to run:
1. Download one of the models given on the main page. Click [here](https://drive.google.com/drive/folders/1Q6BKeEkZyxSGsocK2p_mqgzLwlNvbHFJ?usp=sharing) for the lastest model.
2. Checkout the corresponding commit history or use ```server``` branch if you like to use the latest model.
2. Set the paths and the other options in the file ```server/conf.json```.
3. Run the server ```python server/server.py -c server/conf.json```. (Requires Flask)
4. Go to ```localhost:[given_port]``` and enjoy.

For high quality results, please use the library versions shown in the ```requirements.txt``` file.