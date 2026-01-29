# PostureGuard 🧘‍♂️

AI-powered posture & focus monitor that uses your webcam to detect bad posture and phone distractions. Get reminded by Gandalf or Yoda when you've been slouching for too long!

## Features

- **Posture Detection** - Detects slouching forward and side tilting using MediaPipe Pose
- **Phone Detection** - Spots your phone using YOLOv8 object detection
- **Timed Alerts** - Only alerts after 1 minute of sustained bad behavior (no false alarms!)
- **Fun Reminders** - Random Gandalf/Yoda voice clips to correct you

## Setup

```bash
# Clone the repo
git clone <your-repo-url>
cd postureguard

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Add Sound Files

Add your favorite Gandalf/Yoda MP3 or WAV clips to the `sounds/` folder:

```
sounds/
├── gandalf_shall_not_pass.mp3
├── yoda_do_or_do_not.mp3
└── ...
```

**Where to find clips:** Search YouTube for "Gandalf you shall not pass" or "Yoda quotes" and use a YouTube to MP3 converter, or find sound effect sites.

## Usage

```bash
python main.py
```

- Sit in front of your webcam
- The app shows your pose landmarks and status
- Slouch or grab your phone for 1+ minute → hear a reminder!
- Press **'q'** to quit

## Configuration

Edit these values in `main.py`:

```python
ALERT_THRESHOLD = 60   # seconds before alert (default: 1 min)
ALERT_COOLDOWN = 30    # seconds between alerts
SLOUCH_THRESHOLD = 0.1 # forward lean sensitivity
TILT_THRESHOLD = 0.05  # side tilt sensitivity
```

## How It Works

1. **MediaPipe Pose** tracks 33 body landmarks from your webcam
2. Compares nose position vs shoulders to detect forward slouch
3. Compares left/right shoulder heights to detect side tilt
4. **YOLOv8** runs object detection to find phones in frame
5. Timers track how long issues persist
6. After 60 seconds → plays random sound from `sounds/` folder

## Tech Stack

- OpenCV - Webcam capture
- MediaPipe - Pose estimation
- YOLOv8 (Ultralytics) - Object detection
- playsound - Audio playback

## License

MIT
