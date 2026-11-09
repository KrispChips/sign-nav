# Live demo: Ray-Ban Meta glasses → Android phone → PC

```
glasses ──Bluetooth (DAT)──> phone app (android/) ──Wi-Fi / Tailscale / USB──> PC: main.py
   ▲                              │  frames, head yaw                              │
   └── speech (phone TTS, played ─┘ <──────────── sentences to say ────────────────┘
       on the glasses over Bluetooth)
```

The phone app only relays: it streams the glasses' camera and head motion to the PC and
speaks what the PC sends back. All the work (Grounding DINO, Gemma, depth, OCR) runs on the PC.
The phone is the server for everything, so the PC only needs the phone's address and no
inbound firewall rules.

## Protocol

| what | where | format |
|---|---|---|
| frames | `http://<phone>:8080/frames.mjpg` | MJPEG (`multipart/x-mixed-replace`), newest frame only, `--source` |
| | `http://<phone>:8080/snapshot.jpg`, `http://<phone>:8080/` | one frame / a page to check it in a browser |
| speech | `tcp <phone>:9002`, PC → phone | one JSON line per sentence: `{"text": "...", "kind": "sign"\|"place", "t": <unix s>, "ttl_s": 10\|3}` |
| head yaw | `tcp <phone>:9003`, phone → PC | JSON lines ~30/s: `{"yaw_deg": <deg, + = turning right>, "t": <unix s>}` |

Speech: the phone speaks sentences in order and never interrupts one; a sentence still
waiting `ttl_s` seconds after it **arrived** is dropped (a hallway warning said late is worse
than none). Hallway / entrance / open-area sentences are `place` (3 s), sign reads are `sign` (10 s).
The PC holds sentences while the phone is unreachable and drops them once expired.

Yaw: from the glasses' IMU (DAT Motion, experimental): the gyroscope's rate about the
vertical (from the accelerometer), integrated. It drifts slowly; sign-nav only uses it
relative to its smoothed heading, to pause hallway detection while you glance sideways. If
turning your head right makes the number on the phone go **down**, tick "Invert yaw" in the
app (or pass `--gyro-sign -1` on the PC).

## 1. Build and install the phone app (once)

Needs Android Studio (or just a JDK 17+ and the Android SDK) on the PC, the phone in
developer mode with USB debugging, and the Meta AI app with **Developer Mode** on (Meta AI
app → Settings → your glasses → Developer Mode). Uses Meta's DAT SDK 1.0.1 from Maven Central
(no token needed).

```powershell
cd C:\Users\krith\sign-nav\android
.\gradlew.bat :app:installDebug          # or open the android\ folder in Android Studio and Run
```

With Developer Mode, no app credentials are needed. Without it, register an app in the
Wearables Developer Center and put `mwdat_application_id=...` and `mwdat_client_token=...`
in `android\local.properties` (never commit them). Motion is experimental; if the app logs
"motion unavailable", enable Motion for the app in the Developer Center (streaming works
without it, just no yaw).

## 2. In the app

1. **Connect glasses (Meta AI app)**: registers this app with Meta AI (once).
2. **Start**: asks for the glasses camera permission the first time (in the Meta AI app),
   then starts the stream and the three servers. The status shows the stream size, decoded
   and sent frame rates, KB/s, yaw, the PC clients connected, and the phone's addresses.
3. **Test speech**: you should hear it in the glasses. If it plays on the phone speaker,
   make sure the glasses are the phone's active Bluetooth audio device.

Settings: video quality **High 720×1280** (default) or Medium 504×896; glasses stream fps;
frames sent to the PC per second (default 10) and JPEG quality (default 70). It keeps running
with the screen off (foreground service + notification).

## 3. On the PC

Check the stream first: open `http://<phone>:8080/` in a browser.

```powershell
python main.py --source http://<phone>:8080/frames.mjpg --speak-to <phone>:9002 --imu-from <phone>:9003 --hfov 70 --out-video live.mp4 --show
```

`main.py` waits for the phone and reconnects if the stream or the links drop. Add `--no-tts`
only if you also want silence on the PC: with `--speak-to` the PC never speaks itself.

### Where `<phone>` comes from

- **Same Wi-Fi** (home): the phone's Wi-Fi address shown in the app (e.g. `192.168.1.23`).
- **Tailscale** (phone on campus, PC at home): the phone's Tailscale address (`100.x.y.z`,
  shown in the app on the `tun0` interface, or in the Tailscale app) or its Tailscale name.
  Install Tailscale on both and sign in with the same account. Frames go phone → home over
  the phone's upload: ~50-80 KB per 720p frame at quality 70, so 5 fps ≈ 2-3 Mbit/s
  (≈ 0.5-1 GB per 30-45 min on mobile data); lower "frames sent" or the JPEG quality if the
  upload is slow. Before leaving: PC sleep off, Ollama running, `main.py` started (it waits
  for the phone), or start it remotely (Chrome Remote Desktop / RDP over Tailscale).
- **USB cable** (when a network blocks phone↔PC traffic, e.g. some campus Wi-Fi):
  ```powershell
  adb forward tcp:8080 tcp:8080; adb forward tcp:9002 tcp:9002; adb forward tcp:9003 tcp:9003
  python main.py --source http://localhost:8080/frames.mjpg --speak-to localhost:9002 --imu-from localhost:9003 --hfov 70
  ```

## Without the phone: the fake phone

`tools/fake_phone.py` serves a video file the same way (MJPEG on :8080, prints the sentences
it receives on :9002, sends yaw on :9003), to test the PC side:

```powershell
python tools/fake_phone.py walk.mp4
python main.py --source http://localhost:8080/frames.mjpg --speak-to localhost:9002 --imu-from localhost:9003 --hfov 70 --show
```

## What changes compared with recorded videos

- **Resolution**: at most 720×1280 (Medium: 504×896) vs 1440×1920 recordings. Sign text is
  2-3x fewer pixels, so the legibility trigger waits until you're closer; that's correct
  behaviour, signs are just read later. Hallway detection barely changes (it uses a small
  depth map anyway).
- **Field of view**: the stream may be cropped differently from recordings: check `--hfov`
  (or calibrate with `calibrate.py` on the stream).
- **Head yaw is real now**: hallway detection pauses while your head is turned >25° from
  your walking direction, which removes the false events from looking around.
- **Latency**: Bluetooth + network add a fraction of a second; Gemma's few seconds per sign
  dominate.

## Troubleshooting

| symptom | check |
|---|---|
| browser can't open `http://<phone>:8080/` | same network? Tailscale on in both? campus Wi-Fi blocking device-to-device traffic → USB (`adb forward`) |
| app: "camera would not attach" / no frames | Meta AI app Developer Mode on; glasses connected and unfolded; camera permission granted (Start asks) |
| app: stream size shown but "decoded 0 fps" | the phone's HEVC decoder; send the app's log (logcat tag `SignNavLive`) |
| speech plays on the phone, not the glasses | glasses must be the phone's Bluetooth audio output |
| "motion unavailable" | enable Motion for the app in Wearables Developer Center; everything else works |
| yaw goes the wrong way | "Invert yaw" in the app, or `--gyro-sign -1` on the PC |
| `main.py` prints `[source] waiting for ...` forever | wrong address / port, or the app isn't started |
