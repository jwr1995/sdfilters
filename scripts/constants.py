"""
DATASET PATH CONSTANTS
Included datasets:
- Librispeech (read speech, English)
- MLS (read speech, multilingual)
- AMI (conversational speech, English)
- Alimeeting (conversational speech, Chinese)
- Emilia (in-the-wild speech, multilingual)
- YODAS (in-the-wild speech, multilingual)
- FOR (antispoofing, English)
- ODSS (antispoofing, English)
- MUSAN (noise+music+speech)
- DNC (noise)
- DEMAND (noise)
- DEMAND-chunked (noise, 30s segments)
- FSDNoisy18k (audio events)
- OpenMIC-2018 (music)
"""
import os
from scripts.hf import ACCESS_TOKEN

OUTPUT_DIR='/media/will/DATA/corpora'

# Local annotations directory (in project root)
ANNOTATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'annotations')

SPEECH_DIR=os.path.join(OUTPUT_DIR, 'speech')
NOISE_DIR=os.path.join(OUTPUT_DIR, 'noise')
MUSIC_DIR=os.path.join(OUTPUT_DIR, 'music')

ITW_SPEECH_DIR = os.path.join(SPEECH_DIR, 'itw')
READ_SPEECH_DIR = os.path.join(SPEECH_DIR, 'read')
CONVERSATIONAL_SPEECH_DIR = os.path.join(SPEECH_DIR, 'conversational')

# read speech datasets
LIBRISPEECH_DIR = os.path.join(READ_SPEECH_DIR, 'en', 'librispeech')
MLS_DIR = os.path.join(READ_SPEECH_DIR, 'ml', 'mls')

# conversational speech datasets
AMI_DIR = os.path.join(CONVERSATIONAL_SPEECH_DIR, 'en', 'ami')
ALIMEETING_DIR = os.path.join(CONVERSATIONAL_SPEECH_DIR, 'zh', 'alimeeting')

# in-the-wild speech datasets
EMILIA_DIR = os.path.join(ITW_SPEECH_DIR, 'emilia')
YODAS_DIR = os.path.join(ITW_SPEECH_DIR, 'yodas')

# antispoofing speech datasets
FOR_DIR = os.path.join(READ_SPEECH_DIR, 'en', 'for')
ODSS_DIR = os.path.join(READ_SPEECH_DIR, 'en', 'odss')

# noise datasets
MUSAN_DIR = os.path.join(NOISE_DIR, 'musan')
DNC_DIR = os.path.join(NOISE_DIR, 'dnc')
DEMAND_DIR = os.path.join(NOISE_DIR, 'demand')
DEMAND_CHUNKED_DIR = os.path.join(NOISE_DIR, 'demand-chunked')
FSDNOISY18K_DIR = os.path.join(NOISE_DIR, 'fsdnoisy18k')

# music datasets
OPENMIC_2018_DIR = os.path.join(MUSIC_DIR, 'openmic-2018')