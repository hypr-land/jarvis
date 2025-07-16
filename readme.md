<h1>jarvis CLI</h1>

<p>Jarvis, but it does stuff, can be local, or uses groq API, check jarvis_config.yaml</p>
https://www.youtube.com/watch?v=vCD4DlF2p1c&ab_channel=demo for demo
<p><code>.hyprland.</code> is my discord, if you need anything<p>
<p>also note that jarvis isnt really meant to be a chatbot, but he probably would work as one</p>
<p>mainly because of rate limits, but if your on ollama it shouldnt be a issue</p>

<p>please do NOT flame me for the spaghetti code, it might not be, idk</p>

<p>its a CLI, so yeah, just run <code>!help</code> or <code>!h</code> to see all commands</p>

<h2>features</h2>
<ul>
  <li>Web searching!!</li>
  <li>File management!! just tell jarvis to make a file, or just use the commands built in to do so! he can even write stuff inside of it including code!!!</li>
  <li>git stuff!!! he can push and commit stuff to git!!!</li>
  <li>Speech to text and Text to speech!!!! very cool!!!!</li>
  <li>Also includes wake word detection!!!</li>
  <li>Also rhino (speech to intent) but unused (for people who want to use it, its there)</li>
  <li>See stuff via llama!!! just use spectacle (using a custom shortcut) (or the command "vision (url/path) (prompt") ) (doesnt respond as jarvis for some reason :sob:)</li>
  <li>Open stuff for you!! just ask him to do it (you can change what extension will get opened by what in config)</li>
  <li>also includes command operations, you can pipe, run in background, and so on!</li>
  <li>includes a chrome extension!! nothing special, cant do anything listed above aside from chat and summarize (button is buggy, you need to right click and "summarize with jarvis")</li>
  <li>includes a (bad) server flag!! which disables some of the stuff you wouldnt need if you just wanna have jarvis to make files for you and such (--server) also runs a flask server!</li>
  <li>very customizable, most of it is modularized, and has variables for input/output of things so you can edit it easily, along with decent yaml config, i think?</li>
  <li>Email integration, setup a few env stuff and you can send emails using him</li>
  <li>natural langauge stuff!! just tell him to do something and he will do it, you can say anything</li>
  <li>aliases!!! you can make your aliases to do your commands, works with command operations!</li>
  <li>run !help to see all commands!</li>
</ul>

<p>if you want just the jarvis stuff, just do <code>tts enable ; voice ;</code></p>

<p>Screenshot integration only works on KDE and with spectacle, it should work on any other distro though aside from some kde specific stuff</p>
<ul>
  <li>Windows support is basically impossible, it might work on WSL, but again not tested, STT might not work well</li>
 <li>set desktop_integration_enabled: false and dbus_enabled: false in jarvis_config.yaml for the best chance of it working</li>
</ul>

<h2>to install:</h2>
<ul>
  <li> run the dockerfile i guess, will run it automatically</li>
  <li><code>python jarvis.py afterwards</code></li>
  <li>id also reccomend doing <code>python jarvis.py --test</code> before running it (whether all the things work), but you do you</li>
</ul>

<p>for the other stuff:</p>
<ul>
  <li><a href="https://console.groq.com/keys">https://console.groq.com/keys</a> for the api key for the AI </li>
  <li><a href="https://picovoice.ai/">https://picovoice.ai/</a> for the api key for the wake word + speech to intent</li>
  <li><s>it does use google speech recog for the stt, but no key is required</s> it supports google SR, Groq API via whisper, or local via sphinx now</li>

  <li><you will have to do it manuallly, for the wake word, train it on your voice (i think) and same for speech to intent, you need to do your own connection stuff></li>
</ul>

<hr>

<p>to get the browser extension:</p>
<ul>
  <li>go to <code>chrome://extensions</code></li>
  <li>turn on dev mode</li>
  <li>load unpacked</li>
  <li>choose the chrome_extension folder</li>
</ul>

<p>and it should work, you can add screenshot (via spectacle) by making a shortcut that runs a command and it should handle the rest automatically</p>
<p>do note that jarvis.py needs to be active for it work (because of dbus_handler.py needing to be on)</p>
<pre><code>bus-send --session --type=method_call --dest=org.jarvis.Service /org/jarvis/Service org.jarvis.Service.TriggerScreenshotAnalysis</code></pre>

<p>licensed under apache, to note</p>
<h2>Requirements</h2>
<ul>
  <li>uses about 110MB at most, usually 90-100mb</li>
  <li>0% cpu, at most 0.13% cpu</li>
</ul>
