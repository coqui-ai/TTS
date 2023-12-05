# Mary-TTS API Support for Coqui-TTS

## What is Mary-TTS?

[Mary (Modular Architecture for Research in sYnthesis) Text-to-Speech](http://mary.dfki.de/) is an open-source (GNU LGPL license), multilingual Text-to-Speech Synthesis platform written in Java. It was originally developed as a collaborative project of [DFKI’s](http://www.dfki.de/web) Language Technology Lab and the [Institute of Phonetics](http://www.coli.uni-saarland.de/groups/WB/Phonetics/) at Saarland University, Germany. It is now maintained by the Multimodal Speech Processing Group in the [Cluster of Excellence MMCI](https://www.mmci.uni-saarland.de/) and DFKI.
MaryTTS has been around for a very! long time. Version 3.0 even dates back to 2006, long before Deep Learning was a broadly known term and the last official release was version 5.2 in 2016.
You can check out this OpenVoice-Tech page to learn more: https://openvoice-tech.net/index.php/MaryTTS

## Why Mary-TTS compatibility is relevant

Due to its open-source nature, relatively high quality voices and fast synthetization speed Mary-TTS was a popular choice in the past and many tools implemented API support over the years like screen-readers (NVDA + SpeechHub), smart-home HUBs (openHAB, Home Assistant) or voice assistants (Rhasspy, Mycroft, SEPIA). A compatibility layer for Coqui-TTS will ensure that these tools can use Coqui as a drop-in replacement and get even better voices right away.

## API and code examples

Like Coqui-TTS, Mary-TTS can run as HTTP server to allow access to the API via HTTP GET and POST calls. The best documentations of this API are probably the [web-page](https://github.com/marytts/marytts/tree/master/marytts-runtime/src/main/resources/marytts/server/http), available via your self-hosted Mary-TTS server and the [Java docs page](http://mary.dfki.de/javadoc/marytts/server/http/MaryHttpServer.html).
Mary-TTS offers a larger number of endpoints to load styles, audio effects, examples etc., but compatible tools often only require 3 of them to work:
- `/locales` (GET) - Returns a list of supported locales in the format `[locale]\n...`, for example "en_US" or "de_DE" or simply "en" etc.
- `/voices` (GET) - Returns a list of supported voices in the format `[name] [locale] [gender]\n...`, 'name' can be anything without spaces(!) and 'gender' is traditionally `f` or `m`
- `/process?INPUT_TEXT=[my text]&INPUT_TYPE=TEXT&LOCALE=[locale]&VOICE=[name]&OUTPUT_TYPE=AUDIO&AUDIO=WAVE_FILE` (GET/POST) - Processes the input text and returns a wav file. INPUT_TYPE, OUTPUT_TYPE and AUDIO support additional values, but are usually static in compatible tools.

If your Coqui-TTS server is running on `localhost` using `port` 59125 (for classic Mary-TTS compatibility) you can us the following CURL requests to test the API:

Return locale of active voice, e.g. "en":
```bash
curl http://localhost:59125/locales
```

Return name of active voice, e.g. "glow-tts en u"
```bash
curl http://localhost:59125/voices
```

Create a wav-file with spoken input text:
```bash
curl http://localhost:59125/process?INPUT_TEXT=this+is+a+test > test.wav
```

You can enter the same URLs in your browser and check-out the results there as well.

### How it works and limitations

A classic Mary-TTS server would usually show all installed locales and voices via the corresponding endpoints and accept the parameters `LOCALE` and `VOICE` for processing. For Coqui-TTS we usually start the server with one specific locale and model and thus cannot return all available options. Instead we return the active locale and use the model name as "voice". Since we only have one active model and always want to return a WAV-file, we currently ignore all other processing parameters except `INPUT_TEXT`. Since the gender is not defined for models in Coqui-TTS we always return `u` (undefined).
We think that this is an acceptable compromise, since users are often only interested in one specific voice anyways, but the API might get extended in the future to support multiple languages and voices at the same time.
