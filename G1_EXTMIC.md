# g1_extmic mode: lavalier on the robot, processing on a laptop

A lavalier microphone is plugged into the robot's Jetson through a USB sound card.
`jetson_mic_bridge.py` runs on the Jetson: it sends the lavalier audio to the laptop and
plays the assistant's answers on the robot speaker. STT, LLM and TTS run on the laptop.

```
lavalier → USB sound card → Jetson (jetson_mic_bridge.py)
        ── UDP 5556, PCM 16 kHz ──►  laptop (VAD → Whisper → LLM → TTS)
        ◄── TCP 5557, WAV ────────   answer → g1_audio_play → robot speaker
```

The laptop opens both connections, so a firewall on the laptop does not get in the way and
the Jetson does not need to know the laptop's IP. Everything is signed with a shared key
(HMAC-SHA256), and every subscription and playback request answers a fresh single-use
challenge from the bridge. Another device on the same Wi-Fi can neither take over the
lavalier stream (including by replaying captured packets) nor play sound through the robot.
The audio itself is not encrypted, so use your own router or a cable rather than a shared
network if eavesdropping matters.

## Running it (on the laptop, in the KuzmichAI folder)

1. Plug the lavalier into the Jetson through an **active** USB sound card (one with a chip).
   A passive mini-jack → Type-C adapter will not work.

2. Copy the bridge to the robot and start it. The SSH password is asked once:
   ```bash
   ./deploy_jetson_bridge.sh unitree@192.168.1.103
   ```
   The script prints the capture devices found on the robot and the bridge log. Make sure
   it says `Using USB capture device ...`. On the first run it creates the key
   `~/.config/kuzmich/bridge_token` and copies it to the robot.

3. Start the assistant:
   ```bash
   VOICE_ENGINE_AUDIO=g1_extmic VOICE_ENGINE_ROBOT_IP=192.168.1.103 ./start.sh
   ```
   The log should show `Audio mode: G1_EXTMIC ...` and
   `External mic: receiving audio from ...`.

To make this mode the default without the variable, change the default in `start.sh`:
`export VOICE_ENGINE_AUDIO="${VOICE_ENGINE_AUDIO:-g1_extmic}"`.

## Network

Put the robot and the laptop on your own router (5 GHz, with reserved IPs for both), not
on the network the robot broadcasts and not on the shared exhibition Wi-Fi. If the laptop
sits next to the robot, an Ethernet cable to the robot's port (internal network
192.168.123.x, Jetson usually at `192.168.123.164`) is the most reliable option; use that
address in the commands above.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `No USB capture card found` in the bridge log | List the devices and pick one: `./deploy_jetson_bridge.sh unitree@<ip> --device plughw:2,0` |
| `External mic: no audio from bridge ...` | The bridge is not running or the IP is wrong. Bridge log on the robot: `ssh unitree@<ip> cat /tmp/kuzmich_mic_bridge.log` |
| `dropped ... unauthenticated/stale packets` / `authentication failed` / `Rejected ... subscribe packets` | The keys do not match: run `./deploy_jetson_bridge.sh` again from this laptop |
| `Subscriber switched ...` in the bridge log | Two assistants are connected to the same bridge; stop one of them |
| Lavalier audio arrives but the robot stays silent | Check the `player: ...` line in the bridge log (it should be `g1_audio_play`) and look for `player rc=` errors |
| The robot talks but does not hear | Test the lavalier on the Jetson: `arecord -D plughw:2,0 -f S16_LE -r 16000 -c 1 -d 5 /tmp/t.wav && aplay /tmp/t.wav` |
| The Jetson has a firewall | Open UDP 5556 and TCP 5557 |

Stop the bridge: `./deploy_jetson_bridge.sh unitree@<ip> --stop`.

## Settings (environment variables; defaults are in start.sh)

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_ENGINE_AUDIO` | `local` | `g1_extmic` turns this mode on |
| `VOICE_ENGINE_BRIDGE_HOST` | `VOICE_ENGINE_ROBOT_IP` | address of the Jetson running the bridge |
| `VOICE_ENGINE_BRIDGE_MIC_PORT` / `_PLAY_PORT` | `5556` / `5557` | bridge ports |
| `VOICE_ENGINE_EXTMIC_PLAYER` | `bridge` | where answers are played: `bridge` (robot speaker through the bridge), `g1` (DDS like the g1 mode; needs `unitree_sdk2py` and a cable into the 192.168.123.x network), `local` (laptop speakers, handy for testing the lavalier) |
| `VOICE_ENGINE_BRIDGE_TOKEN_FILE` | `~/.config/kuzmich/bridge_token` | key file |

Bridge options: `python3 jetson_mic_bridge.py --help` (capture device, ports, volume,
network interface for `g1_audio_play`, custom player command).

## What has been tested

Tested without the robot (a second laptop stood in for the Jetson):
- protocol: audio, playback, rejection of a wrong key, forged, tampered and replayed
  packets (including a captured subscription replayed from another address), expired
  challenges, garbage and oversized requests; the bridge survives all of them;
- a full dialogue: a phrase into the "lavalier" → recognition → LLM answer → playback
  through the bridge;
- two machines over Wi-Fi with a firewall enabled on the "laptop";
- capture from a real microphone with `arecord`, USB card auto-detection, Python 3.8;
- the `local` and `g1` modes behave as before.

Not tested, because no robot was available: the real `g1_audio_play` on the Jetson, the
specific USB sound card and lavalier, and the robot's network.
