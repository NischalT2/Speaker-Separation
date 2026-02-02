# Speaker Separation Tool

A Python-based speaker separation tool that takes an audio file containing two speakers and separates them into two distinct audio files. This project combines speaker diarization and source separation techniques to handle both non-overlapping and overlapping speech scenarios.

## Table of Contents

- [Project Overview](#project-overview)
- [How It Was Created](#how-it-was-created)
- [Getting Started](#getting-started)
- [Technical Approach](#technical-approach)
- [Development Journey](#development-journey)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Project Overview

This tool addresses the challenge of separating two speakers from a single audio recording. The solution uses a hybrid approach that combines:

1. **Speaker Diarization** (`pyannote.audio`): Identifies "who spoke when" by analyzing voice characteristics and temporal patterns
2. **Source Separation** (SpeechBrain SepFormer): Separates overlapping speech where both speakers talk simultaneously

The tool processes audio files and outputs two separate WAV files, each containing only one speaker's voice, even when speakers overlap (though not perfect for files containing overlapping voices)

### Key Features

- **Automatic Speaker Detection**: Identifies and separates two speakers automatically
- **Overlapping Speech Handling**: Uses deep learning models to separate overlapping speech
- **Consistent Speaker Assignment**: Maintains speaker identity across all segments
- **Robust Error Handling**: Comprehensive validation and informative error messages
- **Flexible Output**: Customizable output directory

---

## How It Was Created

1. **Initial Research**: Exploring available libraries and approaches for speaker separation
2. **Prototype Development**: Building a basic diarization-based solution
3. **Problem Identification**: Discovering limitations with overlapping speech
4. **Solution Evolution**: Integrating source separation to handle overlapping segments
5. **Refinement**: Fixing bugs, improving consistency, and optimizing performance

The project evolved from a simple diarization-based approach to a hybrid system that handles real-world audio challenges.

---

## Getting Started

This section provides step-by-step instructions for setting up the project on a new computer.

### Prerequisites

Before you begin, make sure that you have:

- **Python 3.8 or higher** installed
  - Check version: `python --version` or `python3 --version`
  - Download from [python.org](https://www.python.org/downloads/) if needed
- **Internet connection** (for downloading models and dependencies)
- **~2GB free disk space** (for models and dependencies)
- **Hugging Face account** (free account is sufficient)
   -Signup here: https://huggingface.co/join

### Step 1: Clone the Repository

```bash
git clone <repository-url>
```

### Step 2: Create a Virtual Environment

Using a virtual environment is **strongly recommended** to avoid dependency conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt when activated.

**Note**: You'll need to reactivate the virtual environment each time you open a new terminal session.

### Step 3: Install Dependencies

With your virtual environment activated, install all required packages:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

This will install:
- `pyannote.audio>=4.0.0` - Speaker diarization pipeline
- `torch>=2.0.0` & `torchaudio>=2.0.0` - Deep learning framework
- `librosa>=0.10.0` - Audio processing
- `soundfile>=0.12.0` - Audio I/O
- `numpy>=1.24.0` - Numerical operations
- `speechbrain>=0.5.0` - Source separation models
- `scipy>=1.9.0` - Scientific computing

**Installation Time**: The first installation may take 10-15 minutes as PyTorch and other large packages are downloaded.

### Step 4: Set Up Hugging Face Account and Token

The project requires Hugging Face access to download pre-trained models.

#### 4.1 Create Hugging Face Account

1. Go to [https://huggingface.co/join](https://huggingface.co/join)
2. Sign up for a free account
3. Verify your email address

#### 4.2 Create Access Token

1. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name it (e.g., "speaker-separation-tool")
4. Select **"Read"** access (sufficient for this project)
5. Click "Generate token"
6. **Copy the token immediately** (you won't be able to see it again!)

#### 4.3 Accept Model Licenses

You need to accept licenses for the following models:

1. **pyannote/speaker-diarization-3.1** (Required)
   - Go to: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Click "Agree and access repository"
   - This is the main diarization model

2. **pyannote/speaker-diarization-community-1** (Required)
   - Go to: [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - Click "Agree and access repository"
   - This is an alternative/updated diarization model that may be used by the pipeline

3. **pyannote/segmentation-3.0** (Required)
   - Go to: [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Click "Agree and access repository"
   - This model is used for speaker segmentation and overlapped speech detection

4. **speechbrain/sepformer-wsj02mix**
   - This is the source separation model
   - This model is public and can be used without agreeing to any licenses or agreements.

#### 4.4 Set Environment Variable

Set your Hugging Face token as an environment variable:

**On Mac/Linux:**
```bash
export HF_TOKEN="your_token_here"

# To make it permanent, add to your shell profile:
# For zsh (default on macOS):
echo 'export HF_TOKEN="your_token_here"' >> ~/.zshrc
source ~/.zshrc

# For bash:
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**On Windows (Command Prompt):**
```cmd
set HF_TOKEN=your_token_here
```

**On Windows (PowerShell):**
```powershell
$env:HF_TOKEN="your_token_here"

# To make it permanent:
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'your_token_here', 'User')
```

**Verify the token is set:**
```bash
# Mac/Linux
echo $HF_TOKEN

# Windows (Command Prompt)
echo %HF_TOKEN%

# Windows (PowerShell)
echo $env:HF_TOKEN
```

### Step 5: Verify Installation

Test that everything is set up correctly:

```bash
# Make sure virtual environment is activated and you're in the project directory
python main.py --help
```

You should see the help message. If you encounter errors, see the [Troubleshooting](#troubleshooting) section.

### Step 6: Run Your First Separation

1. **Prepare a test audio file**:
   - Use a WAV file containing a conversation between two speakers
   - Place it in the project directory or provide the full path

2. **Run the separation**:
   ```bash
   python main.py your_audio_file.wav
   ```

3. **Check the output**:
   - The files with separated voices will be saved in `./output/`:
     - `speaker_1.wav` - First speaker's audio
     - `speaker_2.wav` - Second speaker's audio

**Note**: The first run will download pre-trained models so it might take a little longer. This is a one-time download and models are cached locally in the `pretrained_models/` directory.

---

## Technical Approach

### Architecture Overview

The tool uses a hybrid approachthat combines two techniques:

1. **Diarization for Non-Overlapping Speech**: When speakers don't talk simultaneously, segments are directly extracted from the original audio using diarization timestamps
2. **Source Separation for Overlapping Speech**: When speakers overlap, the mixed audio is processed through a deep learning model to separate the individual voices. However, this part of the program is still not perfect as in the separated files, there may still be some parts of the audio where you can hear the other speaker's voice (although faintly).

### Processing Pipeline

```
Input Audio File
    ↓
[1] Load Audio using librosa
    ↓
[2] Speaker Diarization (pyannote.audio)
    ├─→ Detect speech segments
    ├─→ Extract speaker embeddings
    ├─→ Cluster speakers
    └─→ Assign timestamps to each speaker
    ↓
[3] Detects Overlapping Segments
    ├─→ Non-overlapping segments → Direct extraction
    └─→ Overlapping segments → Source separation
    ↓
[4] Source Separation (SpeechBrain SepFormer)
    ├─→ Resample to model's required sample rate (8kHz)
    ├─→ Separate mixed audio into two sources
    ├─→ Map separated sources to speaker labels
    └─→ Resample back to original sample rate
    ↓
[5] Merge Segments
    ├─→ Combine non-overlapping segments
    ├─→ Add separated overlapping segments
    └─→ Maintain temporal alignment
    ↓
[6] Output Two WAV Files
    ├─→ speaker_1.wav
    └─→ speaker_2.wav
```

### Key Technical Decisions

1. **Hybrid Approach**: Using diarization for clean segments and source separation for overlaps provides the best balance of quality and performance
2. **Consistency Tracking**: Implemented a mapping system to ensure separated sources are consistently assigned to the same speakers across different overlap segments
3. **Sample Rate Handling**: Proper resampling (8kHz for model, 16kHz for output) ensures compatibility with the source separation model (SpeechBrain Sepformer-wsj02mix)
4. **Volume Balancing**: Automatic volume normalization prevents one speaker from being inaudible

### Libraries and Tools

- **pyannote.audio**: State-of-the-art speaker diarization pipeline with pre-trained models
- **SpeechBrain**: Deep learning toolkit providing SepFormer model for source separation
- **librosa**: Audio loading, preprocessing, and resampling
- **soundfile**: Reliable WAV file I/O
- **PyTorch**: Deep learning framework underlying both diarization and separation models
- **numpy**: Efficient and helpful numerical operations for audio processing

---

## Development Journey

This section documents the research, learning process, and problem-solving approach taken during development.

### Initial Research Phase

#### Readings and Documentation

1. **Speaker Diarization Fundamentals**
   - Read: Bredin et al., "pyannote.audio: neural building blocks for speaker diarization" (2020)
      -source: [Paper](https://arxiv.org/abs/1911.01255)
      - Take away: Used this as a very first introduction to the problem of speaker separation. Introduction to pyannote.audio, a deep-learning based, open source toolkit for speaker diarization. Learned about state of the art diarization pipelines and its limitations and also the suggestion of a unified end-to-end neural approach that is more flexible and easier to optimize.
   - Source: [pyannote.audio GitHub](https://github.com/pyannote/pyannote-audio)
   - Source: [huggingface pyannote documentation](https://huggingface.co/pyannote)
      - Explored pyannote and various models that are present for different problems/ situations, specifically explored speaker diarization section to gain a deeper understanding.

2. **Source Separation Research**
   - Read: Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation" (2019)
   - Read: Subakan et al., "Attention is All You Need in Speech Separation" (2021) - SepFormer paper
      - I further skimmed through these readings in order to further understand the problem of speaker separation, learning about the difference between speaker diarization and source separation. Speaker Diarization is when you segment a single audio file into labeled segments to determine "who spoke when" while source separation is when you isolate individual sound sources, such as separating overlapping voices of two different speakers. From these readings, I was able to gain an introduction to the modern processes of speech separation, in which you separate speech by computing a spectogram and then estimate masks for each speaker on that representation, as well as its limitation. It introduced possible improvements that could be made using end-to-end time domain speech separation systems. Furthermore, in the second paper, I learned about the various bottlenecks and limitations that came with ealier speech separation systems and possible solutions and improved models for speech separation such as Sepformer, an RNN free model built entirely on transformers with multi-head self attention. 
   - souce: [Asteroid Documentation](https://asteroid-team.github.io/asteroid/)
   - Source: [SpeechBrain Documentation](https://speechbrain.github.io/)
      - Again, went over some documentations for various speech separation models and tried to understand how they work under the hood. 

3. **Library Documentation**
   - **pyannote.audio**: [Official Documentation](https://github.com/pyannote/pyannote-audio)
     - Studied Pipeline API, model loading, and diarization output format
   - **Asteroid**: [Asteroid Documentation](https://asteroid-team.github.io/asteroid/)
      - Learned how to use pre-trained Asteroid models
   - **SpeechBrain**: [Inference Documentation](https://speechbrain.github.io/tutorial_basics.html)
     - Learned how to use pre-trained SepFormer models
   - **librosa**: [Audio Processing Guide](https://librosa.org/doc/latest/index.html)
     - Understood audio loading, resampling, and format conversion

#### Videos and Tutorials

1. **Speaker Diarization Tutorials**
   - [Pyannnote Speaker Diarization Tutorial](https://www.youtube.com/watch?v=6k-rMcra_4U)

2. **Source Separation Demos**
   - [Asteroid Explanation](https://www.youtube.com/watch?v=imnZxQwuNcg)
   - [SpeechBrain Source Separation](https://www.youtube.com/watch?v=85ey6m8Zrtg)
   - [Audio Source Separation Architype and Demo](https://www.youtube.com/watch?v=Xr7UOWIniCM)

From these few tutorial videos, I was able to gain a basic understanding of how I could import these models and tools into my program and begin working with them with the help of Cursor.

### Initial Implementation

#### Phase 1: Basic Diarization

**Approach**: Started with a simple diarization-based solution using `pyannote.audio`. Using Cursor, I was able to get a basic working program that took in a wav file as an input and returned 2 wav files as an output, in which 1 file contained 1 speaker while the other file contained 2 speakers. I attempted to test with a second test file, in which there were two speakers with overlapping voices; however, this did not go well. 

**Code Structure**:

The initial implementation consisted of three files. The `main.py` file served as the entry point and command-line interface for the application. It handled argument parsing using Python's `argparse` module, validating that the input file exists and is a valid WAV file. When a user provided an audio file path, `main.py` would instantiate the `SpeakerSeparator` class and call its `separate()` method, then handle any errors that occurred during processing.

The `separator.py` file contained the `SpeakerSeparator` class, which was the core processing engine of the application. This class initializes the `pyannote.audio` Pipeline for speaker diarization, which required loading a pre-trained model from Hugging Face. Once initialized, the pipeline processed the input audio file and returned a diarization object containing speaker turns. Each turn consisted of a start time, end time, and speaker label. The diarization results were stored as a list of tuples in the format `[(start_time, end_time, speaker_label)]`, where each tuple represented a segment where a particular speaker was talking. The `SpeakerSeparator` class would then iterate through these segments, extract the corresponding audio portions from the original audio file using the timestamps, and organize them by speaker label into a dictionary structure. Finally, it would combine all segments for each speaker into continuous audio arrays by creating zero-initialized numpy arrays matching the original audio length and placing each speaker's segments at their correct positions, resulting in two separate audio tracks (one per speaker) with silence where each speaker wasn't speaking.

The `audio_utils.py` file provided essential utility functions for audio input/output operations. The `load_audio()` function used `librosa` to read WAV files, automatically converting stereo audio to mono and resampling all audio to a consistent 16kHz sample rate, returning the audio as a 1D numpy array along with the sample rate. The `save_audio()` function handled writing numpy audio arrays back to WAV files using `soundfile`. The `extract_segment()` function extracts specific time segments from the full audio array by converting time-based timestamps (in seconds) to sample indices using the formula `start_sample = int(start_time * sample_rate)` and `end_sample = int(end_time * sample_rate)`, then slicing the audio array accordingly. The `combine_segments()` function took multiple audio segments with their timestamps and assembled them into a single continuous audio array by creating a zero-initialized array of the required total duration and placing each segment at its correct temporal position. These utility functions handled all the low-level audio manipulation, allowing the main separation logic to work with high-level time-based operations rather than dealing with sample indices directly.

### Problem Discovery: Overlapping Speech. Since we had not touched the source separation problem, it was obvious that when there was a wav file with overlapping voices at the same time, it would not extract the output audio files correctly.

#### Research for Solution
1. **Source Separation Techniques**:
   - Using my previous introduction to source separation and various models that could be for this problem, I attempted to proceed forward with Asteroid's model.

### Implementation: Adding Source Separation

#### Phase 2: Hybrid Approach 

**New Approach**: Combine diarization with source separation.

**Model Selection Journey**:

My initial attempt to solve the source separation problem began with Asteroid's Conv-TasNet model, which I had learned about from research papers and tutorials. With Cursor, I integrated Asteroid into the codebase and attempted to use it for separating overlapping speech segments. However, this approach encountered several issues: the model loading process was complex, there were compatibility issues with the library versions, and the separation quality wasn't meeting expectations. The integration proved to be more challenging than anticipated, and the results were not satisfactory for the use case. Some of the problems that I encountered were that the model was not detecting two speakers in the audio file, which resulted in the program only returning 1 single file with both of the speakers talking in the same file.

After the Asteroid attempt didn't work out, I pivoted to SpeechBrain, which had better documentation and more active development. My first choice within SpeechBrain was the `sepformer-wham` model, which operates at 16kHz sample rate - matching the sample rate of our audio processing pipeline. This seemed like a natural fit since it would avoid the need for resampling. I implemented the integration, but quickly discovered that the model wasn't performing as well as expected either. The separation quality was inconsistent, and there were issues with how the model handled the audio input format. Despite the convenience of matching sample rates, the `wham` model wasn't delivering the separation quality needed for the project. Similarly to the Asteroid model, at one point, the model was not detecting two speakers and often returned two very low quality output files with non-separated audio.

After countless hours of debugging, running into errors, and having extended conversations about the issues with Cursor, ChatGPT, Claude, and Gemini, I went to youtube for some help. While going on a rabbit hole about source separation on Youtube, I enocuntered a video that gave a demo/tutorial on source separation using SpeechBrain's `sepformer-wsj02mix` model [SpeechBrain Source Separation](https://www.youtube.com/watch?v=85ey6m8Zrtg). Due to the fact that this model operates at 8kHz, I was required me to implement a resampling logic to convert the audio from 16kHz to 8kHz before processing, and then resampling back to 16kHz after separation. This is because this model accepted only 8kHz audio. Though this added an extra step in the processing pipeline, the `wsj02mix` model provided significantly better separation quality and more reliable results. The trade-off of adding resampling was worth it for the improved performance, and this model became the final choice for the source separation implementation. However, the audio separation in the returned files from this implementation is still not perfect as the returned files still do not have separate the audios completely. There is still room for the program to be cleaned up.

**Implementation Steps**:

The `source_separator.py` file was created to contain all source separation functionality in a `SourceSeparator` class. This class handles the complete cycle of source separation, from model initialization to audio processing. The `__init__` method sets up the device (CPU or GPU), initializes the model name (defaulting to `speechbrain/sepformer-wsj02mix`), and calls the `_load_model()` method to load the pre-trained SepFormer model from Hugging Face. The `_load_model()` method contains critical compatibility patches, including a workaround for the `huggingface_hub` API changes (converting `use_auth_token` to `token`) and creating a dummy `custom.py` file to prevent download errors. The core `separate()` method performs the actual source separation: it first resamples the input audio from 16kHz to 8kHz (required by the wsj02mix model), converts the numpy array to a PyTorch tensor, runs the model inference to separate the mixed audio into two sources, and then resamples both separated sources back to the original 16kHz sample rate. Additionally, the method implements volume balancing logic that compares the energy levels of the two separated sources and boosts the quieter source (up to 3x) to ensure both speakers are audible, followed by normalization to prevent clipping. The class also includes a `_fallback_separation()` method that provides a basic fallback if the model fails to load, though this produces poor quality results.

The `separator.py` file underwent a lot of changes. The `SpeakerSeparator` class was enhanced to initialize a `SourceSeparator` instance when `use_source_separation=True`, allowing the system to fall back if source separation is unavailable. The main `separate()` method now implements a two-path processing approach: after running diarization and extracting all speaker segments, it calls the `detect_overlapping_segments()` function from `audio_utils.py` to identify which segments have overlapping speech. The function returns two lists: `overlapping_intervals` (time intervals in which multiple speakers are active) and `non_overlapping_segments_list` (intervals in which only one speaker is talking). For non-overlapping segments, the original direct extraction method is used - segments are extracted from the original audio and organized by speaker label. For overlapping intervals, the code extracts the mixed audio segment, passes it to the `SourceSeparator.separate()` method, and receives two separated audio sources. In this case, the challenge is to map these separated sources back to the correct speaker labels from diarization, which is handled by the `_map_separated_sources_to_speakers()` method. It uses correlation analysis between the separated sources and adjacent non-overlapping segments to determine which source corresponds to which speaker. The method also implements consistency tracking using a `source_speaker_mapping` dictionary that stores the mapping for each speaker pair, ensuring that once a mapping is established for a pair of speakers, it's reused for subsequent overlapping segments to prevent issues where halfway through the returned output file, the speaker focus is swapped to the other speaker. Finally, the code merges both non-overlapping and separated overlapping segments into final audio arrays for each speaker, ensuring proper temporal alignment and bounds checking to prevent array index errors.

The `audio_utils.py` file was improved with the `detect_overlapping_segments()` function, which is crucial for identifying when multiple speakers are talking simultaneously. This function takes a list of diarization segments (each as a tuple of `(start_time, end_time, speaker_label)`) and performs a sweep-line algorithm to detect overlaps. It creates event points for each segment's start and end times, sorts them chronologically, and tracks which speakers are active at each moment. When two or more speakers are active simultaneously, it identifies this as an overlapping interval and records the start and end times along with the list of active speakers. The function also handles partial overlaps, so if a segment partially overlaps with an overlap interval, it splits the segment to preserve the non-overlapping portions. The non-overlapping parts are extracted directly from the original audio, while only the truly overlapping portions are processed through source separation. The function returns two lists: `overlapping_intervals` containing tuples of `(start, end, [speaker1, speaker2, ...])` for time periods where multiple speakers are active, and `non_overlapping_segments` containing the segments where only one speaker is talking.

### Major Problems that I encountered

#### Problem 1: Model Loading Errors

**Symptom**: `404 Client Error: Entry Not Found for url: .../custom.py`

**Root Cause**: After a lot of debugging and reading the terminal for the error message and also reading the SpeechBrain source code and documentation, I found the root cause of the problem to be the fact that SpeechBrain was trying to download a `custom.py` file that didn't exist for the model. Though some models require this custom.py, this model did not so we traced the error to be the model's hyperparams.yaml configuration. 

**Solution**: After notifying cursor of this error, it suggested and helped in creating a dummy custom.py file if it didn't exist. 

```python
# Create a dummy custom.py file if it doesn't exist
os.makedirs(savedir, exist_ok=True)
custom_py_path = os.path.join(savedir, "custom.py")
if not os.path.exists(custom_py_path):
    with open(custom_py_path, 'w') as f:
        f.write("# Empty custom.py - not required for this model\n")
```

#### Problem 2: Sample Rate Mismatch

**Symptom**: Model errors or poor separation quality.

**Root Cause**: After a small conversation with cursor about SepFormer, I realized that the SepFormer models have specific sample rate requirements (8kHz for wsj02mix, 16kHz for wham). So, I was able to get this problem out of the way, which resulted in the model error that I was encountering at the time, allowing me to progress further (but again run into other bugs).

**Solution**:
```python
# Resample to model's required sample rate (8kHz for wsj02mix)
target_rate = 8000
if sample_rate != target_rate:
    mixed_audio_resampled = librosa.resample(mixed_audio, orig_sr=sample_rate, target_sr=target_rate)
# ... process with model ...
# Resample back to original sample rate
if sample_rate != target_rate:
    source1 = librosa.resample(source1_8k, orig_sr=target_rate, target_sr=sample_rate)
```


#### Problem 3: Inconsistent Speaker Assignment

**Symptom**: In the output files, speakers would "flip" mid way through the audio, so Speaker 1's voice would suddenly appear in Speaker 2's file.

**Root Cause**: SpeechBrain SepFormer doesn't guarantee consistent ordering of separated sources across different calls. Each overlap segment was processed independently, and source1/source2 could map to different speakers in different segments. So, through logging, I was able to discover that the correlation-based mapping was inconsistent and that the model returns sources in arbitrary order. So, through help of cursor and debugging, I was able to implement a consistency tracking system, in hiwch the program used a dictionary that acted as a memory for speaker assignments. The tuples in this dictionary recorded which speaker corresponds with which separated source from the first time we processed an overlap for that speaker pair. In practice, when a subsequent overlapping segment appears between two speakers who already have overlapped before, the system checks if this speaker pair already exists in the mapping dictionary and if so instead of recalculating the mapping (possibly resulting in a different assignment), it retrieves the stored mapping and applies it directly. 

**Solution**: Implemented consistency tracking:
```python
# Track source-to-speaker mapping for consistency across overlaps
source_speaker_mapping = {}  # Maps (speaker1, speaker2) to (source1_speaker, source2_speaker)

# On first overlap for a speaker pair, determine mapping
# On subsequent overlaps, reuse the same mapping
if speaker_pair in source_speaker_mapping:
    existing_mapping = source_speaker_mapping[speaker_pair]
    # Use existing mapping to maintain consistency
```


#### Problem 4: Missing Speech Segments

**Symptom**: Some parts of the audio where a speaker was clearly talking were missing from the output files. 

**Root Cause**: Segments that partially overlapped with overlap intervals were being completely excluded instead of being split. From debugging logs with the help of Cursor, I was able to find that segments partially overlapping with overlap intervals were being lost, so to solve this, we modified the segment splitting logic to check if the segment was fully overlapping or partially overlapping and if there segments in which there were no overlaps, we would split the segment to preserve non-overlapping parts. This made it so that these intervals were not lost. 

**Solution**: Modified segment splitting logic:
```python
# Split segments that partially overlap to preserve non-overlapping portions
for start, end, speaker in sorted_segments:
    # Check if segment is fully contained within overlap
    # If partially overlapping, split to preserve non-overlapping parts
    if partially_overlapping:
        # Extract non-overlapping portions before and after the overlap
        non_overlapping_segments.append((start, overlap_start, speaker))
        non_overlapping_segments.append((overlap_end, end, speaker))
```


#### Problem 5: Volume Imbalance

**Symptom**: One speaker was much louder than the other in the output, making one speaker inaudible.

**Root Cause**: Source separation models don't preserve original volume levels, and one source might be significantly quieter. Thus, through more conversing with Cursor, looking at possible solutions from Google searches and various sources, and adding energy analysis to compare source volumes, we wre able to discover the large energy differences between separated sources. To fix this, we were able to add scripts to balance the volumes by boosting the source audio to be closer to the louder source audio.

**Solution**: Implemented volume balancing:
```python
# Balance volumes: boost quieter source to be closer to louder source
source1_energy = np.sum(np.abs(source1))
source2_energy = np.sum(np.abs(source2))

if source2_energy > 0 and source1_energy > 0:
    energy_ratio = source1_energy / source2_energy
    if energy_ratio > 2.0:
        # Source1 is much louder, boost source2
        boost_factor = min(energy_ratio / 2.0, 3.0)  # Cap at 3x
        source2 = source2 * boost_factor
```

#### Problem 6: PyTorch Serialization Security

**Symptom**: `RuntimeError: Unknown type for <class 'pyannote.audio.core.task.Problem'>`

**Root Cause**: PyTorch 2.6+ added security restrictions on what classes can be loaded from model files. I was able to find this through the PyTorch documentation and release notes about serialization security. After digging a little bit more on Google and having more conversations with Gemini, I discovered the `add_safe_globals` API, which allowed me to identify classes that needed to be whitelisted so that the program could run.

**Solution**:
```python
from pyannote.audio.core.task import Problem, Resolution, Specifications
torch.serialization.add_safe_globals([
    Problem, 
    Resolution, 
    Specifications, 
    torch.torch_version.TorchVersion,
    np.core.multiarray.scalar,
    np.dtype
])
```

---

## Project Structure

```
Luel Technical/
├── main.py                  # Entry point, CLI interface
├── separator.py            # Core separation logic with diarization
├── source_separator.py     # Source separation using SpeechBrain SepFormer
├── audio_utils.py          # Audio I/O, preprocessing, and utility functions
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── output/                 # Output directory (created automatically)
│   ├── speaker_1.wav
│   └── speaker_2.wav
└── pretrained_models/      # Cached models (created automatically on first run)
    ├── pyannote_speaker-diarization-3.1/
    └── speechbrain_sepformer-wsj02mix/
```

### Code Organization

- **`main.py`**: 
  - Command-line argument parsing
  - Error handling and user feedback
  - Orchestrates the separation process

- **`separator.py`**: 
  - `SpeakerSeparator` class: Main orchestration class
  - Diarization pipeline initialization
  - Overlap detection and classification
  - Segment merging and output generation
  - Consistency tracking for speaker assignment

- **`source_separator.py`**: 
  - `SourceSeparator` class: Handles source separation
  - Model loading and management
  - Audio resampling for model compatibility
  - Volume balancing and normalization

- **`audio_utils.py`**: 
  - `load_audio()`: Load WAV files with librosa
  - `save_audio()`: Save audio to WAV files
  - `extract_segment()`: Extract time segments from audio
  - `detect_overlapping_segments()`: Identify overlapping speech segments
  - `combine_segments()`: Merge multiple segments into continuous audio

---

## Usage

### Basic Usage

```bash
python main.py input_audio.wav
```

This will create two files in the `./output` directory:
- `speaker_1.wav` - First speaker's audio
- `speaker_2.wav` - Second speaker's audio

### Custom Output Directory

```bash
python main.py input_audio.wav --output-dir ./my_results
```

### Get Help

```bash
python main.py --help
```

### Example

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run separation
python main.py test_conversation.wav

# Output files will be in ./output/
ls output/
# speaker_1.wav  speaker_2.wav
```

---

## Troubleshooting

### "HF_TOKEN environment variable not set"

**Problem**: The Hugging Face token is not set or not accessible.

**Solution**:
1. Verify the token is set: `echo $HF_TOKEN` (Mac/Linux) or `echo %HF_TOKEN%` (Windows)
2. Make sure you activated your virtual environment if you set the token there
3. Re-export the token in your current terminal session
4. For permanent setup, add to your shell profile (`.zshrc`, `.bashrc`, or Windows environment variables)

### "Failed to load diarization model"

**Problem**: Cannot download or load the pyannote.audio model.

**Solutions**:
1. **Check license acceptance**: Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and click "Agree and access repository"
2. **Verify token**: Ensure your `HF_TOKEN` has read access
3. **Check internet connection**: Models download on first use (~500MB)
4. **Clear cache**: Delete `~/.cache/huggingface/` and try again
5. **Check disk space**: Ensure you have at least 2GB free space

### "No speakers detected"

**Problem**: Diarization didn't detect any speakers in the audio.

**Solutions**:
1. Verify the audio file contains speech (not just music or silence)
2. Check audio file is not corrupted - try opening it in an audio player
3. Try a different audio file to isolate the issue
4. Check audio format - should be WAV format
5. Verify audio has reasonable volume levels

### "Only 1 speaker detected"

**Problem**: Diarization only found one speaker despite expecting two.

**Possible Causes**:
1. Audio actually has only one speaker
2. Speakers have very similar voices
3. Poor audio quality makes distinction difficult
4. One speaker dominates the conversation

**What the Tool Does**:
- The tool will attempt full-file source separation as a fallback
- This may still produce two outputs, but quality may vary

### Import Errors

**Problem**: `ModuleNotFoundError` or `ImportError` when running.

**Solutions**:
1. **Activate virtual environment**: Make sure `(venv)` appears in your prompt
2. **Reinstall dependencies**: `pip install -r requirements.txt --force-reinstall`
3. **Check Python version**: `python --version` should be 3.8 or higher
4. **Upgrade pip**: `pip install --upgrade pip` then reinstall requirements

### GPU/CPU Issues

**Problem**: Slow processing or GPU not being used.

**Solutions**:
1. **GPU detection**: The tool automatically uses GPU if available
2. **CPU is fine**: CPU processing works but is slower (acceptable for most use cases)
3. **First run is slow**: Models download and initialize on first run
4. **Check PyTorch CUDA**: `python -c "import torch; print(torch.cuda.is_available())"` to verify GPU support

### Virtual Environment Issues

**Problem**: Virtual environment not working or commands not found.

**Solutions**:
1. **Recreate virtual environment**: 
   ```bash
   rm -rf venv
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   ```
2. **Windows activation**: Use `venv\Scripts\activate.bat` if `.ps1` doesn't work
3. **Check activation**: You should see `(venv)` in your prompt
4. **Reinstall in venv**: Make sure to install requirements after activating venv

### Model Download Issues

**Problem**: Models fail to download or download is very slow.

**Solutions**:
1. **Check internet connection**: Models are ~500MB total
2. **Use VPN if needed**: Some networks block Hugging Face
3. **Retry**: Models are cached, so failed downloads can be retried
4. **Manual download**: Models are cached in `~/.cache/huggingface/` - you can check there

### Audio Quality Issues

**Problem**: Output audio has artifacts, static, or poor quality.

**Solutions**:
1. **Input quality matters**: Use clear, high-quality input audio
2. **Check sample rate**: Tool handles resampling automatically
3. **Volume issues**: Tool includes automatic volume balancing
4. **Overlapping speech**: Some artifacts are expected in heavily overlapping segments

---

## References

### Research Papers

1. **Speaker Diarization**:
   - Bredin, H., et al. (2020). "pyannote.audio: neural building blocks for speaker diarization." *ICASSP 2020*

2. **Source Separation**:
   - Luo, Y., & Mesgarani, N. (2019). "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation." *ICASSP 2019*
   - Subakan, C., et al. (2021). "Attention is All You Need in Speech Separation." *ICASSP 2021* (SepFormer)

### Documentation

1. **pyannote.audio**:
   - GitHub: [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
   - Documentation: [https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)

2. **SpeechBrain**:
   - Website: [https://speechbrain.github.io/](https://speechbrain.github.io/)
   - GitHub: [https://github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)
   - Inference Tutorial: [https://speechbrain.github.io/tutorial_basics.html](https://speechbrain.github.io/tutorial_basics.html)

3. **librosa**:
   - Documentation: [https://librosa.org/doc/latest/index.html](https://librosa.org/doc/latest/index.html)

4. **Hugging Face Models**:
   - pyannote/speaker-diarization-3.1: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - pyannote/speaker-diarization-community-1: [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - pyannote/segmentation-3.0: [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - speechbrain/sepformer-wsj02mix: [https://huggingface.co/speechbrain/sepformer-wsj02mix](https://huggingface.co/speechbrain/sepformer-wsj02mix)

### Additional Resources

- **PyTorch Documentation**: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **NumPy Documentation**: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
- **Python Virtual Environments**: [https://docs.python.org/3/tutorial/venv.html](https://docs.python.org/3/tutorial/venv.html)

---

## License

The underlying libraries and models have their own licenses:

- **pyannote.audio**: MIT License
- **SpeechBrain**: Apache 2.0 License
- **librosa**: ISC License
- **PyTorch**: BSD-style License

Please refer to each library's license for details.

---
