# Luel Final Stage Technical Interview Assignment: Speaker Separation Tool

## Overview

Welcome! For this assignment, we'd like you to build a simple **speaker separation tool** — a program that takes an audio file with two people talking and splits it into two separate audio files, one for each speaker.

Think of it like this: Imagine you have a recording of a podcast with two hosts. Your tool should be able to separate their voices so you end up with two files — one where you only hear Host A, and another where you only hear Host B.

---

## What You'll Build

**Input:** A `.wav` audio file containing two people speaking (could be a conversation, interview, etc.)

**Output:** Two separate `.wav` files:
- `speaker_1.wav` — containing only the first speaker's voice
- `speaker_2.wav` — containing only the second speaker's voice

---

## Requirements

1. **Accept a WAV file as input** (you can hardcode the path or accept it as a command-line argument)
2. **Process the audio** to identify and separate the two speakers
3. **Output two separate WAV files**, one per speaker
4. **Include a README** explaining:
   - How to run your solution
   - What libraries/tools you used and why
   - Any challenges you faced and how you solved them

---

## Guidelines & Encouragement

### Use Existing Tools & Libraries
You are **strongly encouraged** to use existing libraries and pre-trained models! This is not about reinventing the wheel — it's about knowing how to find and use the right tools. Some popular options include:

- **Python libraries:** `pyannote.audio`, `speechbrain`, `spleeter`, `asteroid`, `resemblyzer`
- **Pre-trained models:** Many open-source models exist for speaker diarization and source separation
- **Audio processing:** `librosa`, `pydub`, `soundfile`, `scipy`

### Read Research & Documentation
Feel free to look up:
- Research papers on speaker separation and source separation
- GitHub repositories, Huggingface projects, etc. existing implementations, and improve upon existing ones
- Blog posts and tutorials explaining the concepts

### Use AI Coding Assistants (Vibecoding!)
We **encourage** you to use AI tools like ChatGPT, Claude, GitHub Copilot, or Cursor to help you:
- Understand concepts you're unfamiliar with
- Debug errors
- Write boilerplate code
- Explore different approaches

This is how modern developers work! Just make sure you understand what the code does.

---

## Evaluation Criteria

We're looking for:

| Criteria | What We're Looking For |
|----------|------------------------|
| **Problem-Solving** | How did you approach the problem? What decisions did you make? |
| **Resourcefulness** | Did you find and use appropriate existing tools/libraries? |
| **Code Quality** | Is your code readable and reasonably organized? |
| **Documentation** | Can we understand how to run your solution and what it does? |
| **Honesty** | Be upfront about what works, what doesn't, and what you'd improve |

---

## Tips for Success

1. **Start simple** — Get something working first, then improve it
2. **Google is your friend** — Search for "speaker separation Python", "source separation deep learning", etc.
3. **Don't panic if it's not perfect** — Real-world audio separation is a hard problem. We want to see your approach, not a flawless result
4. **Document your journey** — Tell us what you tried, what worked, what didn't

---

## Submission

Please submit:
- [ ] Your source code (Python preferred, but other languages are acceptable)
- [ ] A `README.md` file with setup and usage instructions
- [ ] A brief write-up of your approach (can be in the README)

You can submit as a GitHub repository link or a zip file.

---

## Time Expectation

This assignment is designed to take approximately **3-48 hours** of focused work. If you find yourself spending significantly more time, it's okay to submit a partial solution with notes on what you would do next.

**Suggested deadline:** 1 week from receiving this assignment. That said, work at your own pace — life happens, and we'd rather see thoughtful work than rushed work. Just keep us updated if you need more time.

---

## Questions?

If anything is unclear, please don't hesitate to reach out. Asking good questions is a valuable skill!

Good luck, and have fun with it! 🎧