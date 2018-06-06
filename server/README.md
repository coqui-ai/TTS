## TTS example web-server
Steps to run:
1. Download one of the models given on the main page.
2. Checkout the corresponding commit history. 
2. Set paths and other options in the file ```server/conf.json```.
3. Run the server ```python server/server.py -c conf.json```. (Requires Flask)
4. Go to ```localhost:[given_port]``` and enjoy.

Note that the audio quality on browser is slightly worse due to the encoder quantization. 