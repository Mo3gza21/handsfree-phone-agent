# The phone app

Android only. iOS doesn't let one app control others, so hands-free control like this
isn't possible there.

## What it does

1. **Listens** (foreground service + Android speech recognition, offline when the phone has the
   language pack). Say the wake word then the command: "Maya, call Mom" / "مايا، اتصل بماما".
2. **Reads the screen** through an accessibility service and turns it into the same compact text
   the model was trained on (`[3] button "Send"`).
3. **Asks the model** (llama.cpp, on the phone's CPU) for one action. A grammar forces the model
   to output a valid action every time.
4. **Acts**: tap, type, scroll, back, home, open app; then reads the screen again, until the
   model says `done`. It speaks the result out loud, and asks you out loud when a request is
   ambiguous ("Which Ahmed: Ahmed Work or Ahmed Gym?").

Nothing leaves the phone.

## Build

1. Install [Android Studio](https://developer.android.com/studio). In **SDK Manager → SDK Tools**,
   tick **NDK (Side by side) 29.0.13113456** and **CMake 3.31.6**.
2. Fetch llama.cpp (pinned to the same version the export script uses). On Windows, use Git Bash:
   ```bash
   ./android/setup_llama.sh
   ```
3. Open the `android/` folder in Android Studio, plug in the phone (USB debugging on), press **Run**.
   The first build compiles llama.cpp and takes a few minutes.

## Set up on the phone

Open **Hands-Free Agent**:

1. **Turn on accessibility.** Find "Hands-Free Agent" in the list and turn it on. On Android 13+,
   if it's greyed out: Settings → Apps → Hands-Free Agent → ⋮ → *Allow restricted settings* first.
2. **Allow microphone + notifications.**
3. **Choose the model file:** the `phone-agent-q4_k_m.gguf` from Colab (copy it to the phone's
   Downloads first). It's copied into the app once.
4. Pick the language (Arabic (Egypt) / English) and the wake word. Leave the wake word empty to
   treat everything you say as a command.
5. **Start hands-free.** A notification shows it's listening; it has a Stop button.

Use the typed **Run** box to try commands without voice. The app goes to the background and the
agent works on whatever screen is under it. Step-by-step logs are in Logcat under `AgentLoop`.

## Notes

- Speed: on a mid-range phone, expect roughly 1-3 seconds per step with the 1.5B model. The
  system prompt is cached between steps, so only the new screen is processed each time.
- Opening apps by name uses a `<queries>` entry for launcher apps in the manifest, so no
  special permission is needed.
- If the model gets stuck it stops after repeating the same step 3 times, or after 25 steps.
